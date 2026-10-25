package ai.auralis.turbograph;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

import ai.auralis.turbograph.Methods.Method;
import ai.auralis.turbograph.Methods.Query;
import ai.auralis.turbograph.Methods.Result;
import ai.auralis.turbograph.TwoBitCodec.Code;

/**
 * TurboVec-HippoRAG: one structure that physically combines the TurboVec 2-bit index
 * with HippoRAG's knowledge graph.
 * <p>
 * Nodes: passages [0, P), entities [P, P+E), facts [P+E, P+E+F). Every node is a
 * 2-bit TurboQuant code. Edges are exact (int target, type, weight):
 * <ul>
 * <li>CONTAINS passage-entity (HippoRAG), weight 1;</li>
 * <li>RELATION entity-entity for every fact subject-object (HippoRAG), weight = #facts;</li>
 * <li>SYNONYM entity-entity with cosine >= 0.8 among 2-bit nearest neighbours (HippoRAG), weight = cosine;</li>
 * <li>ARG fact-entity and SOURCE fact-passage: facts are nodes of the same graph;</li>
 * <li>SIMILAR passage-passage navigable links (TurboVec graph).</li>
 * </ul>
 * Full-precision vectors are kept beside the structure only for the float HippoRAG 2
 * baseline and for optional exact re-scoring; they are not part of its size.
 */
final class TurboHippo {

    static final int CONTAINS = 0;
    static final int RELATION = 1;
    static final int SYNONYM = 2;
    static final int ARG = 3;
    static final int SOURCE = 4;
    static final int SIMILAR = 5;
    static final String[] TYPES = {"CONTAINS", "RELATION", "SYNONYM", "ARG", "SOURCE", "SIMILAR"};

    final TwoBitCodec codec = new TwoBitCodec(768, 0x5EEDL);
    final int P;
    final int E;
    final int F;
    final Code[] code;
    final float[][] full;
    final int[][] adj;
    final byte[][] type;
    final float[][] weight;
    /** Number of passages containing each entity (for HippoRAG node specificity). */
    final int[] entityPassages;
    final int[] factSubject;
    final int[] factObject;
    final long buildMs;

    private TurboHippo(int p, int e, int f, Code[] code, float[][] full, int[][] adj, byte[][] type, float[][] weight,
                       int[] entityPassages, int[] factSubject, int[] factObject, long buildMs) {
        this.P = p;
        this.E = e;
        this.F = f;
        this.code = code;
        this.full = full;
        this.adj = adj;
        this.type = type;
        this.weight = weight;
        this.entityPassages = entityPassages;
        this.factSubject = factSubject;
        this.factObject = factObject;
        this.buildMs = buildMs;
    }

    int size() {
        return P + E + F;
    }

    long edges() {
        long e = 0;
        for (int[] a : adj) {
            e += a.length;
        }
        return e;
    }

    /** 2-bit codes (D/4 + 4 bytes) + edges (4-byte target, 1-byte type, 2-byte fp16 weight). */
    long bytes2bit() {
        return (long) size() * (codec.paddedDim() / 4 + 4) + edges() * 7;
    }

    /** The same graph with float32 vectors, as HippoRAG stores it. */
    long bytesFloat() {
        return (long) size() * 768 * 4 + edges() * 8;
    }

    static TurboHippo build(Path dir, String ds, float[][] passages) throws Exception {
        long t0 = System.currentTimeMillis();
        int p = passages.length;
        List<String> entityNames = Files.readAllLines(dir.resolve(ds + ".entities.txt")).stream()
                .filter(s -> !s.isEmpty()).toList();
        float[][] ent = Corpus.readF16(dir.resolve(ds + ".entities.f16"), 768);
        List<String> factLines = Files.readAllLines(dir.resolve(ds + ".facts.tsv")).stream()
                .filter(s -> !s.isEmpty()).toList();
        float[][] fac = Corpus.readF16(dir.resolve(ds + ".facts.f16"), 768);
        int e = Math.min(entityNames.size(), ent.length);
        int f = Math.min(factLines.size(), fac.length);
        Map<String, Integer> entIdx = new HashMap<>();
        for (int i = 0; i < e; i++) {
            entIdx.put(entityNames.get(i), p + i);
        }
        int n = p + e + f;
        float[][] full = new float[n][];
        System.arraycopy(passages, 0, full, 0, p);
        System.arraycopy(ent, 0, full, p, e);
        System.arraycopy(fac, 0, full, p + e, f);
        TwoBitCodec codec = new TwoBitCodec(768, 0x5EEDL);
        Code[] code = new Code[n];
        IntStream.range(0, n).parallel().forEach(i -> code[i] = codec.encode(full[i]));

        List<Map<Integer, float[]>> nb = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            nb.add(new HashMap<>());
        }
        int[] factSubject = new int[f];
        int[] factObject = new int[f];
        Map<Integer, Set<Integer>> passEnts = new HashMap<>();
        for (int i = 0; i < f; i++) {
            String[] t = factLines.get(i).split("\t", -1);
            int pid = Integer.parseInt(t[0]);
            Integer s = entIdx.get(t[1]);
            Integer o = entIdx.get(t[3]);
            int fid = p + e + i;
            factSubject[i] = s == null ? -1 : s;
            factObject[i] = o == null ? -1 : o;
            if (pid >= 0 && pid < p) {
                edge(nb, fid, pid, SOURCE, 1f, false);
            }
            for (Integer x : new Integer[] {s, o}) {
                if (x != null) {
                    edge(nb, fid, x, ARG, 1f, false);
                    if (pid >= 0 && pid < p) {
                        passEnts.computeIfAbsent(pid, k -> new HashSet<>()).add(x);
                    }
                }
            }
            if (s != null && o != null && !s.equals(o)) {
                edge(nb, s, o, RELATION, 1f, true);
            }
        }
        int[] entityPassages = new int[e];
        passEnts.forEach((pid, ents) -> {
            for (int x : ents) {
                edge(nb, pid, x, CONTAINS, 1f, false);
                entityPassages[x - p]++;
            }
        });

        // SYNONYM: 2-bit nearest neighbours among entities, kept if exact cosine >= 0.8.
        int[][] entNear = nsw(codec, code, p, p + e, 10, 20, 48);
        for (int i = 0; i < e; i++) {
            int u = p + i;
            for (int v : entNear[i]) {
                if (v > u) {
                    double c = Corpus.cos(full[u], full[v]);
                    if (c >= 0.8) {
                        edge(nb, u, v, SYNONYM, (float) c, false);
                    }
                }
            }
        }
        // SIMILAR: navigable links between passages (TurboVec graph).
        int[][] passNear = nsw(codec, code, 0, p, 12, 24, 48);
        for (int i = 0; i < p; i++) {
            for (int v : passNear[i]) {
                edge(nb, i, v, SIMILAR, 1f, false);
            }
        }

        int[][] adj = new int[n][];
        byte[][] type = new byte[n][];
        float[][] weight = new float[n][];
        for (int i = 0; i < n; i++) {
            Map<Integer, float[]> m = nb.get(i);
            adj[i] = new int[m.size()];
            type[i] = new byte[m.size()];
            weight[i] = new float[m.size()];
            int j = 0;
            for (var en : m.entrySet()) {
                adj[i][j] = en.getKey();
                type[i][j] = (byte) en.getValue()[0];
                weight[i][j] = en.getValue()[1];
                j++;
            }
        }
        return new TurboHippo(p, e, f, code, full, adj, type, weight, entityPassages, factSubject, factObject,
                System.currentTimeMillis() - t0);
    }

    /** Undirected edge; RELATION edges accumulate weight (number of facts linking the pair). */
    private static void edge(List<Map<Integer, float[]>> nb, int a, int b, int t, float w, boolean accumulate) {
        for (int[] pair : new int[][] {{a, b}, {b, a}}) {
            float[] cur = nb.get(pair[0]).get(pair[1]);
            if (cur == null) {
                nb.get(pair[0]).put(pair[1], new float[] {t, w});
            } else if (accumulate && cur[0] == t) {
                cur[1] += w;
            } else if (t < cur[0]) {
                cur[0] = t;
                cur[1] = w;
            }
        }
    }

    /** Navigable small world over nodes [lo, hi) with 2-bit beam insertion (parallel batches). */
    private static int[][] nsw(TwoBitCodec codec, Code[] code, int lo, int hi, int m, int mMax, int ef) {
        int n = hi - lo;
        if (n <= 1) {
            return new int[Math.max(0, n)][0];
        }
        Code[] sub = Arrays.copyOfRange(code, lo, hi);
        int[] order = IntStream.range(0, n).toArray();
        SplittableRandom rnd = new SplittableRandom(42);
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        int[][] sim = new int[n][];
        int[] len = new int[n];
        int entry = order[0];
        sim[entry] = new int[mMax + 1];
        int k = 1;
        while (k < n) {
            int batch = k < 1000 ? 1 : Math.min(256, n - k);
            final int from = k;
            int[][] found = new int[batch][];
            IntStream.range(0, batch).parallel().forEach(b -> {
                int v = order[from + b];
                found[b] = TurboGraph.beam(codec, codec.decodeRotated(sub[v]), entry, sub, sim, len, ef, m);
            });
            for (int b = 0; b < batch; b++) {
                int v = order[from + b];
                sim[v] = new int[mMax + 1];
                for (int u : found[b]) {
                    sim[v][len[v]++] = u;
                    sim[u][len[u]++] = v;
                    if (len[u] > mMax) {
                        TurboGraph.prune(codec, u, sub, sim, len, mMax);
                    }
                }
            }
            k += batch;
        }
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = new int[len[i]];
            for (int j = 0; j < len[i]; j++) {
                out[i][j] = sim[i][j] + lo;
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ retrieval

    /** How a TurboHippo method scores vectors and walks the graph. */
    record Config(String name, boolean twoBit, boolean localPush, boolean factNodes, boolean similarEdges,
                  double lambda) {
    }

    /**
     * HippoRAG 2 retrieval on this structure:
     * <ol>
     * <li>score all facts against the query (2-bit scan, or float) and take the top 5;</li>
     * <li>seed their entities with fact score / #passages containing the entity, and seed
     * passages with 0.05 x their normalised dense score;</li>
     * <li>Personalized PageRank with damping 0.5 over the exact edges (global power iteration
     * like HippoRAG, or local push), and rank passages by PageRank mass (or fused with the
     * passage's own score when lambda > 0).</li>
     * </ol>
     */
    Method method(Config cfg) {
        TurboHippo g = this;
        return new Method() {
            public String name() { return cfg.name(); }
            public boolean fullPrecision() { return !cfg.twoBit(); }
            public long indexBytes() { return cfg.twoBit() ? g.bytes2bit() : g.bytesFloat(); }
            public Result run(Query q) {
                long[] scored = {0};
                float[] ps = new float[g.P];
                float pMax = Float.NEGATIVE_INFINITY;
                float pMin = Float.POSITIVE_INFINITY;
                for (int i = 0; i < g.P; i++) {
                    ps[i] = g.score(q, i, cfg.twoBit());
                    pMax = Math.max(pMax, ps[i]);
                    pMin = Math.min(pMin, ps[i]);
                }
                scored[0] += g.P;
                int[] topFacts = Methods.topK(factScores(q, cfg.twoBit(), scored), 5);
                Map<Integer, Double> seed = new HashMap<>();
                Map<Integer, Integer> seen = new HashMap<>();
                for (int fi : topFacts) {
                    double fs = g.score(q, g.P + g.E + fi, cfg.twoBit());
                    for (int ent : new int[] {g.factSubject[fi], g.factObject[fi]}) {
                        if (ent >= 0) {
                            double spec = Math.max(1, g.entityPassages[ent - g.P]);
                            seed.merge(ent, fs / spec, Double::sum);
                            seen.merge(ent, 1, Integer::sum);
                        }
                    }
                }
                seen.forEach((ent, c) -> seed.computeIfPresent(ent, (k, v) -> v / c));
                if (seed.isEmpty()) {
                    return new Result(Methods.topK(ps, 100), scored[0], -1);
                }
                float range = Math.max(1e-6f, pMax - pMin);
                int[] passSeeds = cfg.localPush() ? Methods.topK(ps, 200) : IntStream.range(0, g.P).toArray();
                for (int i : passSeeds) {
                    seed.merge(i, 0.05 * (ps[i] - pMin) / range, Double::sum);
                }
                Map<Integer, Double> mass = cfg.localPush() ? g.pushPpr(seed, 0.5, 1e-5, cfg)
                        : g.globalPpr(seed, 0.5, 30, cfg);
                float[] out = new float[g.P];
                Arrays.fill(out, Float.NEGATIVE_INFINITY);
                double maxM = mass.values().stream().mapToDouble(Double::doubleValue).max().orElse(1);
                int touched = 0;
                for (var en : mass.entrySet()) {
                    int v = en.getKey();
                    if (v < g.P) {
                        out[v] = (float) (cfg.lambda() > 0 ? ps[v] + cfg.lambda() * en.getValue() / maxM : en.getValue());
                        touched++;
                    }
                }
                return new Result(Methods.topK(out, 100), scored[0], cfg.localPush() ? touched : -1);
            }
        };
    }

    private float[] factScores(Query q, boolean twoBit, long[] scored) {
        float[] s = new float[F];
        for (int i = 0; i < F; i++) {
            s[i] = score(q, P + E + i, twoBit);
        }
        scored[0] += F;
        return s;
    }

    float score(Query q, int node, boolean twoBit) {
        return twoBit ? codec.scoreCorrected(q.rq(), code[node]) : (float) Corpus.cos(q.emb(), full[node]);
    }

    private boolean usable(int u, int j, Config cfg) {
        int t = type[u][j];
        int v = adj[u][j];
        if (t == SIMILAR && !cfg.similarEdges()) {
            return false;
        }
        return cfg.factNodes() || (t != ARG && t != SOURCE && v < P + E && u < P + E);
    }

    /** HippoRAG-style PageRank: power iteration over the whole graph. */
    private Map<Integer, Double> globalPpr(Map<Integer, Double> seed, double damping, int iters, Config cfg) {
        int n = size();
        double[] s = new double[n];
        double tot = seed.values().stream().mapToDouble(Double::doubleValue).sum();
        seed.forEach((k, v) -> s[k] = v / tot);
        double[] outW = new double[n];
        for (int u = 0; u < n; u++) {
            for (int j = 0; j < adj[u].length; j++) {
                if (usable(u, j, cfg)) {
                    outW[u] += weight[u][j];
                }
            }
        }
        double[] p = s.clone();
        for (int it = 0; it < iters; it++) {
            double[] nx = new double[n];
            for (int u = 0; u < n; u++) {
                if (p[u] == 0 || outW[u] == 0) {
                    continue;
                }
                double share = damping * p[u] / outW[u];
                for (int j = 0; j < adj[u].length; j++) {
                    if (usable(u, j, cfg)) {
                        nx[adj[u][j]] += share * weight[u][j];
                    }
                }
            }
            for (int u = 0; u < n; u++) {
                nx[u] += (1 - damping) * s[u];
            }
            p = nx;
        }
        Map<Integer, Double> out = new HashMap<>();
        for (int u = 0; u < P; u++) {
            if (p[u] > 0) {
                out.put(u, p[u]);
            }
        }
        return out;
    }

    /** Local push PageRank (Andersen-Chung-Lang) over the same edges; touches only nearby nodes. */
    private Map<Integer, Double> pushPpr(Map<Integer, Double> seed, double damping, double eps, Config cfg) {
        double alpha = 1 - damping;
        double tot = seed.values().stream().mapToDouble(Double::doubleValue).sum();
        Map<Integer, Double> p = new HashMap<>();
        Map<Integer, Double> r = new HashMap<>();
        seed.forEach((k, v) -> r.put(k, v / tot));
        ArrayDeque<Integer> queue = new ArrayDeque<>(r.keySet());
        BitSet queued = new BitSet(size());
        r.keySet().forEach(queued::set);
        long pushes = 0;
        while (!queue.isEmpty() && pushes < 500_000) {
            int u = queue.poll();
            queued.clear(u);
            double ru = r.getOrDefault(u, 0.0);
            double wSum = 0;
            int deg = 0;
            for (int j = 0; j < adj[u].length; j++) {
                if (usable(u, j, cfg)) {
                    wSum += weight[u][j];
                    deg++;
                }
            }
            if (deg == 0) {
                p.merge(u, ru, Double::sum);
                r.put(u, 0.0);
                continue;
            }
            if (ru < eps * deg) {
                continue;
            }
            pushes++;
            p.merge(u, alpha * ru, Double::sum);
            r.put(u, 0.0);
            for (int j = 0; j < adj[u].length; j++) {
                if (!usable(u, j, cfg)) {
                    continue;
                }
                int v = adj[u][j];
                double rv = r.merge(v, (1 - alpha) * ru * weight[u][j] / wSum, Double::sum);
                if (!queued.get(v) && rv >= eps * Math.max(1, adj[v].length)) {
                    queued.set(v);
                    queue.add(v);
                }
            }
        }
        Map<Integer, Double> out = new HashMap<>();
        p.forEach((k, v) -> {
            if (k < P) {
                out.put(k, v);
            }
        });
        return out;
    }
}
