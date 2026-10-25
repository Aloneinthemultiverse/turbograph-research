package ai.auralis.turbograph;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/** Every retrieval idea behind one interface, so they run side by side on the same queries. */
public final class Methods {

    private Methods() {
    }

    /** A query: its text, full embedding (for exact methods) and rotated form (for 2-bit methods). */
    record Query(String text, float[] emb, float[] rq) {
    }

    /**
     * What a method returns: chunks best first, how many nodes it scored, and how many
     * chunks its (k-free) traversal accepted, or -1 for top-k methods.
     */
    record Result(int[] ranked, long scored, int accepted) {
    }

    interface Method {
        String name();

        /** True if the method reads full-precision embeddings at query time. */
        boolean fullPrecision();

        /** Bytes of the index the method searches. */
        long indexBytes();

        Result run(Query q);
    }

    // ---------------------------------------------------------------- baselines

    /** Exact cosine over full embeddings: the ground truth / upper bound. */
    static Method exact(Corpus c) {
        return new Method() {
            public String name() { return "Exact float32 scan"; }
            public boolean fullPrecision() { return true; }
            public long indexBytes() { return (long) c.n * c.emb[0].length * 4; }
            public Result run(Query q) {
                return new Result(topByExact(c, q, allChunks(c.n), 100), c.n, -1);
            }
        };
    }

    /** TurboVec flat scan, 2-bit only (no full-precision anywhere). */
    static Method flat2bit(TurboGraph g) {
        return new Method() {
            public String name() { return "TurboVec flat 2-bit scan"; }
            public boolean fullPrecision() { return false; }
            public long indexBytes() { return (long) g.chunks * (g.codec.paddedDim() / 4 + 4); }
            public Result run(Query q) {
                float[] s = new float[g.chunks];
                for (int i = 0; i < g.chunks; i++) {
                    s[i] = g.codec.scoreCorrected(q.rq(), g.code[i]);
                }
                return new Result(topK(s, 100), g.chunks, -1);
            }
        };
    }

    /** TurboVec as Auralis uses it: 2-bit scan, then exact re-score of a shortlist. */
    static Method flat2bitRescore(Corpus c, TurboGraph g, int shortlist) {
        return new Method() {
            public String name() { return "TurboVec flat 2-bit + exact rescore(" + shortlist + ")"; }
            public boolean fullPrecision() { return true; }
            public long indexBytes() { return (long) g.chunks * (g.codec.paddedDim() / 4 + 4); }
            public Result run(Query q) {
                float[] s = new float[g.chunks];
                for (int i = 0; i < g.chunks; i++) {
                    s[i] = g.codec.score(q.rq(), g.code[i]);
                }
                return new Result(topByExact(c, q, topK(s, shortlist), 100), g.chunks, -1);
            }
        };
    }

    /** BM25 over chunk text: the lexical leg of hybrid retrieval. */
    static Method bm25(Corpus c) {
        Map<String, int[]> postings = new HashMap<>();
        Map<String, List<int[]>> build = new HashMap<>();
        int[] len = new int[c.n];
        long total = 0;
        for (int i = 0; i < c.n; i++) {
            Map<String, Integer> tf = new HashMap<>();
            for (String t : tokens(c.text[i])) {
                tf.merge(t, 1, Integer::sum);
                len[i]++;
            }
            total += len[i];
            final int doc = i;
            tf.forEach((t, f) -> build.computeIfAbsent(t, x -> new ArrayList<>()).add(new int[] {doc, f}));
        }
        long[] bytes = {0};
        build.forEach((t, l) -> {
            int[] flat = new int[l.size() * 2];
            for (int j = 0; j < l.size(); j++) {
                flat[2 * j] = l.get(j)[0];
                flat[2 * j + 1] = l.get(j)[1];
            }
            postings.put(t, flat);
            bytes[0] += flat.length * 4L;
        });
        double avg = (double) total / c.n;
        return new Method() {
            public String name() { return "BM25 keyword"; }
            public boolean fullPrecision() { return false; }
            public long indexBytes() { return bytes[0]; }
            public Result run(Query q) {
                float[] s = new float[c.n];
                long touched = 0;
                for (String t : new LinkedHashSet<>(tokens(q.text()))) {
                    int[] p = postings.get(t);
                    if (p == null) {
                        continue;
                    }
                    int df = p.length / 2;
                    double idf = Math.log(1 + (c.n - df + 0.5) / (df + 0.5));
                    for (int j = 0; j < p.length; j += 2) {
                        int d = p[j];
                        int f = p[j + 1];
                        s[d] += (float) (idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len[d] / avg)));
                        touched++;
                    }
                }
                return new Result(topK(s, 100), touched, -1);
            }
        };
    }

    // ---------------------------------------------------------- graph methods

    /** Classic KG retrieval: link query entities, take their 1-hop chunks, rank by 2-bit score. */
    static Method kHop(TurboGraph g) {
        return new Method() {
            public String name() { return "KG entity link + 1-hop"; }
            public boolean fullPrecision() { return false; }
            public long indexBytes() { return g.bytes(); }
            public Result run(Query q) {
                Set<Integer> cand = new LinkedHashSet<>();
                for (int t : queryTerms(g, q.text())) {
                    for (int v : g.adj[t]) {
                        if (g.kind[v] == TurboGraph.Kind.CHUNK) {
                            cand.add(v);
                        }
                    }
                }
                float[] s = new float[g.chunks];
                Arrays.fill(s, Float.NEGATIVE_INFINITY);
                for (int v : cand) {
                    s[v] = g.codec.scoreCorrected(q.rq(), g.code[v]);
                }
                int[] r = topK(s, Math.min(100, cand.size()));
                return new Result(r, cand.size(), cand.size());
            }
        };
    }

    /** Best-first beam walk over all edges with 2-bit scores (top-k style, width ef). */
    static Method beamWalk(TurboGraph g, int ef) {
        return new Method() {
            public String name() { return "2-bit beam walk (ef=" + ef + ")"; }
            public boolean fullPrecision() { return false; }
            public long indexBytes() { return g.bytes(); }
            public Result run(Query q) {
                long[] scored = {0};
                BitSet seen = new BitSet(g.size());
                PriorityQueue<float[]> cand = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));
                PriorityQueue<float[]> best = new PriorityQueue<>((a, b) -> Float.compare(a[0], b[0]));
                List<Integer> starts = new ArrayList<>(queryTerms(g, q.text()));
                starts.add(g.entry);
                for (int st : starts) {
                    seen.set(st);
                    float s = g.codec.score(q.rq(), g.code[st]);
                    scored[0]++;
                    cand.add(new float[] {s, st});
                    best.add(new float[] {s, st});
                }
                Map<Integer, Float> chunkScore = new HashMap<>();
                while (!cand.isEmpty()) {
                    float[] c = cand.poll();
                    if (best.size() >= ef && c[0] < best.peek()[0]) {
                        break;
                    }
                    int u = (int) c[1];
                    int[] nb = g.adj[u];
                    int step = Math.max(1, nb.length / 256);
                    for (int j = 0; j < nb.length; j += step) {
                        int v = nb[j];
                        if (seen.get(v)) {
                            continue;
                        }
                        seen.set(v);
                        float s = g.codec.score(q.rq(), g.code[v]);
                        scored[0]++;
                        if (g.kind[v] == TurboGraph.Kind.CHUNK) {
                            chunkScore.put(v, s);
                        }
                        if (best.size() < ef || s > best.peek()[0]) {
                            cand.add(new float[] {s, v});
                            best.add(new float[] {s, v});
                            if (best.size() > ef) {
                                best.poll();
                            }
                        }
                    }
                }
                return new Result(rankMap(chunkScore, 100), scored[0], -1);
            }
        };
    }

    /**
     * k-free threshold expansion: greedy descent to local peaks, then accept every
     * neighbour within delta of the best 2-bit score seen, expanding accepted nodes.
     */
    static Method thresholdExpansion(TurboGraph g, float delta) {
        return new Method() {
            public String name() { return "2-bit threshold expansion (delta=" + delta + ")"; }
            public boolean fullPrecision() { return false; }
            public long indexBytes() { return g.bytes(); }
            public Result run(Query q) {
                long[] scored = {0};
                List<Integer> starts = new ArrayList<>(queryTerms(g, q.text()));
                starts.add(g.entry);
                List<Integer> peaks = descend(g, q.rq(), starts, scored);
                Map<Integer, Float> acc = new HashMap<>();
                BitSet seen = new BitSet(g.size());
                ArrayDeque<Integer> queue = new ArrayDeque<>();
                float best = Float.NEGATIVE_INFINITY;
                for (int p : peaks) {
                    float s = g.codec.score(q.rq(), g.code[p]);
                    scored[0]++;
                    best = Math.max(best, s);
                    seen.set(p);
                    acc.put(p, s);
                    queue.add(p);
                }
                while (!queue.isEmpty() && acc.size() < 20_000) {
                    int u = queue.poll();
                    if (acc.get(u) < best - delta) {
                        continue;
                    }
                    int[] nb = g.adj[u];
                    int step = Math.max(1, nb.length / 256);
                    for (int j = 0; j < nb.length; j += step) {
                        int v = nb[j];
                        if (seen.get(v)) {
                            continue;
                        }
                        seen.set(v);
                        float s = g.codec.score(q.rq(), g.code[v]);
                        scored[0]++;
                        if (s >= best - delta) {
                            acc.put(v, s);
                            queue.add(v);
                            best = Math.max(best, s);
                        }
                    }
                }
                float cut = best - delta;
                Map<Integer, Float> chunks = new HashMap<>();
                acc.forEach((v, s) -> {
                    if (s >= cut && g.kind[v] == TurboGraph.Kind.CHUNK) {
                        chunks.put(v, s);
                    }
                });
                return new Result(rankMap(chunks, 100), scored[0], chunks.size());
            }
        };
    }

    /** Seeds for personalized PageRank. */
    enum Seeds {
        /** Query term nodes + greedy-descent peaks: fully local, O(neighbours). */
        LOCAL,
        /** Query term nodes + chunks within 0.05 of the best 2-bit score in a flat scan. */
        SCAN
    }

    /**
     * Local push personalized PageRank (Andersen-Chung-Lang) over the exact edges.
     * Relevance mass flows from the seeds; a node pushes only while its residual
     * exceeds eps * degree, so work stays local. Edge weights = relation weight x
     * 2-bit relevance of the neighbour^gamma x term idf.
     */
    static Method pushPpr(TurboGraph g, Seeds seeds, double alpha, double eps, double gamma) {
        return pushPpr(g, seeds, alpha, eps, gamma, 0);
    }

    /**
     * Push-PPR with a fused ranking when {@code lambda > 0}: every chunk the push touched
     * is ranked by its own 2-bit relevance plus lambda x its PageRank mass (normalised by
     * the largest), so a direct match is not buried by graph mass and graph-connected
     * evidence still rises. lambda = 0 ranks by PageRank mass alone.
     */
    static Method pushPpr(TurboGraph g, Seeds seeds, double alpha, double eps, double gamma, double lambda) {
        double[] relW = {1.0, 0.3, 0.7, 1.0};
        return new Method() {
            public String name() {
                return (lambda > 0 ? "Fused 2-bit + Push-PPR " : "Push-PPR ") + seeds + " (a=" + alpha
                        + ", eps=" + eps + ", g=" + gamma + (lambda > 0 ? ", lambda=" + lambda : "") + ")";
            }
            public boolean fullPrecision() { return false; }
            public long indexBytes() { return g.bytes(); }
            public Result run(Query q) {
                long[] scored = {0};
                Map<Integer, Double> seed = new HashMap<>();
                for (int t : queryTerms(g, q.text())) {
                    seed.merge(t, (double) g.idf[t], Double::sum);
                }
                if (seeds == Seeds.LOCAL) {
                    List<Integer> starts = new ArrayList<>(seed.keySet());
                    starts.add(g.entry);
                    for (int p : descend(g, q.rq(), starts, scored)) {
                        seed.merge(p, 2.0, Double::sum);
                    }
                } else {
                    float[] s = new float[g.chunks];
                    float best = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < g.chunks; i++) {
                        s[i] = g.codec.score(q.rq(), g.code[i]);
                        best = Math.max(best, s[i]);
                    }
                    scored[0] += g.chunks;
                    for (int i = 0; i < g.chunks; i++) {
                        if (s[i] >= best - 0.05f) {
                            seed.merge(i, Math.exp((s[i] - best) * 40), Double::sum);
                        }
                    }
                }
                double total = seed.values().stream().mapToDouble(Double::doubleValue).sum();
                Map<Integer, Double> p = new HashMap<>();
                Map<Integer, Double> r = new HashMap<>();
                seed.forEach((k, v) -> r.put(k, v / total));
                Map<Integer, Float> rel = new HashMap<>();
                ArrayDeque<Integer> queue = new ArrayDeque<>(r.keySet());
                BitSet queued = new BitSet(g.size());
                r.keySet().forEach(queued::set);
                long pushes = 0;
                while (!queue.isEmpty() && pushes < 200_000) {
                    int u = queue.poll();
                    queued.clear(u);
                    int[] nb = g.adj[u];
                    int step = Math.max(1, nb.length / 256);
                    int deg = (nb.length + step - 1) / step;
                    double ru = r.getOrDefault(u, 0.0);
                    if (deg == 0 || ru < eps * deg) {
                        if (deg == 0) {
                            p.merge(u, ru, Double::sum);
                            r.put(u, 0.0);
                        }
                        continue;
                    }
                    pushes++;
                    p.merge(u, alpha * ru, Double::sum);
                    r.put(u, 0.0);
                    double[] w = new double[deg];
                    double wSum = 0;
                    for (int j = 0, k = 0; j < nb.length; j += step, k++) {
                        int v = nb[j];
                        Float sv = rel.get(v);
                        if (sv == null) {
                            sv = g.codec.score(q.rq(), g.code[v]);
                            rel.put(v, sv);
                            scored[0]++;
                        }
                        double base = Math.max(0.01, sv);
                        w[k] = relW[g.rel[u][j]] * Math.pow(base, gamma)
                                * (g.kind[v] == TurboGraph.Kind.TERM ? g.idf[v] : 1.0);
                        wSum += w[k];
                    }
                    for (int j = 0, k = 0; j < nb.length; j += step, k++) {
                        int v = nb[j];
                        double add = (1 - alpha) * ru * w[k] / wSum;
                        double rv = r.merge(v, add, Double::sum);
                        int vDeg = Math.min(g.adj[v].length, 256);
                        if (!queued.get(v) && rv >= eps * Math.max(1, vDeg)) {
                            queued.set(v);
                            queue.add(v);
                        }
                    }
                }
                Map<Integer, Float> chunks = new HashMap<>();
                if (lambda > 0) {
                    double maxP = p.values().stream().mapToDouble(Double::doubleValue).max().orElse(1);
                    Set<Integer> cand = new LinkedHashSet<>(rel.keySet());
                    cand.addAll(p.keySet());
                    for (int v : cand) {
                        if (g.kind[v] != TurboGraph.Kind.CHUNK) {
                            continue;
                        }
                        Float sv = rel.get(v);
                        if (sv == null) {
                            sv = g.codec.score(q.rq(), g.code[v]);
                            scored[0]++;
                        }
                        chunks.put(v, (float) (sv + lambda * p.getOrDefault(v, 0.0) / maxP));
                    }
                } else {
                    p.forEach((v, mass) -> {
                        if (g.kind[v] == TurboGraph.Kind.CHUNK && mass > 0) {
                            chunks.put(v, mass.floatValue());
                        }
                    });
                }
                return new Result(rankMap(chunks, 100), scored[0], chunks.size());
            }
        };
    }

    /** Re-ranks another method's candidates by exact cosine (needs full precision). */
    static Method rescored(Method inner, Corpus c) {
        return new Method() {
            public String name() { return inner.name() + " + exact rescore"; }
            public boolean fullPrecision() { return true; }
            public long indexBytes() { return inner.indexBytes(); }
            public Result run(Query q) {
                Result r = inner.run(q);
                return new Result(topByExact(c, q, r.ranked(), 100), r.scored(), r.accepted());
            }
        };
    }

    /** Reciprocal rank fusion of several methods (hybrid retrieval). */
    static Method rrf(String name, List<Method> parts) {
        return new Method() {
            public String name() { return name; }
            public boolean fullPrecision() { return parts.stream().anyMatch(Method::fullPrecision); }
            public long indexBytes() { return parts.stream().mapToLong(Method::indexBytes).sum(); }
            public Result run(Query q) {
                Map<Integer, Float> s = new HashMap<>();
                long scored = 0;
                for (Method m : parts) {
                    Result r = m.run(q);
                    scored += r.scored();
                    for (int i = 0; i < r.ranked().length; i++) {
                        s.merge(r.ranked()[i], 1f / (60 + i + 1), Float::sum);
                    }
                }
                return new Result(rankMap(s, 100), scored, -1);
            }
        };
    }

    // ---------------------------------------------------------------- helpers

    static List<Integer> queryTerms(TurboGraph g, String text) {
        List<Integer> out = new ArrayList<>();
        for (String e : g.queryEntities.apply(text)) {
            Integer t = g.termNode.get(e);
            if (t != null && !out.contains(t)) {
                out.add(t);
            }
        }
        for (String tok : text.toUpperCase().split("[^A-Z0-9_-]+")) {
            Integer t = g.termNode.get(tok);
            if (t != null && !out.contains(t)) {
                out.add(t);
            }
        }
        return out;
    }

    /** Greedy descent from each start to a local 2-bit peak. */
    static List<Integer> descend(TurboGraph g, float[] rq, List<Integer> starts, long[] scored) {
        Set<Integer> peaks = new LinkedHashSet<>();
        for (int start : starts) {
            int u = start;
            float su = g.codec.score(rq, g.code[u]);
            scored[0]++;
            for (int guard = 0; guard < 64; guard++) {
                int bestV = -1;
                float bestS = su;
                int[] nb = g.adj[u];
                int step = Math.max(1, nb.length / 256);
                for (int j = 0; j < nb.length; j += step) {
                    float s = g.codec.score(rq, g.code[nb[j]]);
                    scored[0]++;
                    if (s > bestS) {
                        bestS = s;
                        bestV = nb[j];
                    }
                }
                if (bestV < 0) {
                    break;
                }
                u = bestV;
                su = bestS;
            }
            peaks.add(u);
        }
        return new ArrayList<>(peaks);
    }

    static List<String> tokens(String text) {
        List<String> out = new ArrayList<>();
        for (String t : text.toLowerCase().split("[^a-z0-9]+")) {
            if (t.length() > 1) {
                out.add(t);
            }
        }
        return out;
    }

    private static int[] allChunks(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        return a;
    }

    static int[] topByExact(Corpus c, Query q, int[] cand, int k) {
        float[] s = new float[cand.length];
        for (int i = 0; i < cand.length; i++) {
            s[i] = (float) Corpus.cos(q.emb(), c.emb[cand[i]]);
        }
        int[] order = topK(s, Math.min(k, cand.length));
        int[] out = new int[order.length];
        for (int i = 0; i < order.length; i++) {
            out[i] = cand[order[i]];
        }
        return out;
    }

    static int[] topK(float[] s, int k) {
        PriorityQueue<Integer> h = new PriorityQueue<>((a, b) -> Float.compare(s[a], s[b]));
        for (int i = 0; i < s.length; i++) {
            if (s[i] == Float.NEGATIVE_INFINITY) {
                continue;
            }
            if (h.size() < k) {
                h.add(i);
            } else if (s[i] > s[h.peek()]) {
                h.poll();
                h.add(i);
            }
        }
        int[] out = new int[h.size()];
        for (int i = out.length - 1; i >= 0; i--) {
            out[i] = h.poll();
        }
        return out;
    }

    static int[] rankMap(Map<Integer, Float> s, int k) {
        return s.entrySet().stream().sorted((a, b) -> Float.compare(b.getValue(), a.getValue()))
                .limit(k).mapToInt(Map.Entry::getKey).toArray();
    }
}
