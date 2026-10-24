package ai.auralis.turbograph;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ai.auralis.turbograph.Methods.Method;
import ai.auralis.turbograph.Methods.Query;
import ai.auralis.turbograph.Methods.Result;

/**
 * Runs every retrieval method side by side on two query sets built from the corpus.
 * <ul>
 * <li><b>Single-hop</b>: a sentence span cut from a random chunk. Truth: the source
 * chunk, and the exact top-10 by cosine (fidelity to full precision).</li>
 * <li><b>Two-hop</b>: two chunks from different documents that share a rare term;
 * the query joins a span of each. Truth: both chunks (HotpotQA-style supporting
 * evidence). Retrieving both needs the method to cover two facets at once.</li>
 * </ul>
 * Writes results/results.md and results/results.csv.
 */
public final class Bench {

    record Single(Query q, int source, int[] exactTop10) {
    }

    record Pair(Query q, int a, int b, String term) {
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        int nQueries = args.length > 0 ? Integer.parseInt(args[0]) : 100;
        Path out = Path.of("results");
        Files.createDirectories(out.resolve("cache"));
        log("loading corpus from Postgres");
        Corpus c = Corpus.load("jdbc:postgresql://localhost:5433/auralis", "auralis", "auralis");
        log("corpus: %d chunks, %d documents".formatted(c.n, c.docTitle.length));
        TurboGraph g = TurboGraph.build(c, 12, 24, 48);
        log("TurboVec knowledge graph: %d nodes (%d chunks, %d terms, %d docs), %d edges, %.1f MB, built in %d ms"
                .formatted(g.size(), g.chunks, g.terms, g.docs, g.edges(), g.bytes() / 1e6, g.buildMs));
        Corpus.Embedder emb = new Corpus.Embedder("http://localhost:11434", out.resolve("cache/queries.tsv"));

        Random rnd = new Random(7);
        List<Single> singles = new ArrayList<>();
        Method exact = Methods.exact(c);
        while (singles.size() < nQueries) {
            int src = rnd.nextInt(c.n);
            if (c.text[src].length() < 300) {
                continue;
            }
            String text = span(c.text[src], 160);
            float[] e = emb.query(text);
            Query q = new Query(text, e, g.codec.rotate(e));
            int[] top = exact.run(q).ranked();
            singles.add(new Single(q, src, java.util.Arrays.copyOf(top, Math.min(10, top.length))));
        }
        List<Pair> pairs = twoHop(c, g, emb, rnd, nQueries);
        log("queries: %d single-hop, %d two-hop".formatted(singles.size(), pairs.size()));

        List<Method> methods = new ArrayList<>();
        methods.add(exact);
        methods.add(Methods.flat2bit(g));
        Method flatRescore = Methods.flat2bitRescore(c, g, 100);
        methods.add(flatRescore);
        Method bm25 = Methods.bm25(c);
        methods.add(bm25);
        methods.add(Methods.kHop(g));
        methods.add(Methods.beamWalk(g, 200));
        methods.add(Methods.thresholdExpansion(g, 0.08f));
        methods.add(Methods.thresholdExpansion(g, 0.12f));
        Method pprLocal = Methods.pushPpr(g, Methods.Seeds.LOCAL, 0.15, 1e-4, 4);
        Method pprScan = Methods.pushPpr(g, Methods.Seeds.SCAN, 0.15, 1e-4, 4);
        methods.add(pprLocal);
        methods.add(Methods.pushPpr(g, Methods.Seeds.LOCAL, 0.15, 1e-5, 4));
        methods.add(Methods.pushPpr(g, Methods.Seeds.LOCAL, 0.15, 1e-4, 1));
        methods.add(pprScan);
        methods.add(Methods.rescored(pprLocal, c));
        methods.add(Methods.rescored(pprScan, c));
        methods.add(Methods.rrf("Hybrid RRF: flat 2-bit + BM25", List.of(Methods.flat2bit(g), bm25)));
        methods.add(Methods.rrf("Hybrid RRF: flat 2-bit + BM25 + Push-PPR SCAN",
                List.of(Methods.flat2bit(g), bm25, pprScan)));
        methods.add(Methods.rrf("Hybrid RRF: flat+rescore + BM25 + Push-PPR SCAN (Auralis-style)",
                List.of(flatRescore, bm25, pprScan)));

        StringBuilder md = new StringBuilder();
        StringBuilder csv = new StringBuilder(
                "method,full_precision,index_mb,s_recall10_vs_exact,s_source_at1,s_source_at10,s_mrr,"
                        + "t_both_at10,t_both_at20,t_any_at10,ms_per_query,nodes_scored,accepted\n");
        md.append("# TurboVec knowledge graph — side-by-side retrieval benchmark\n\n")
                .append("Corpus: %d chunks, %d documents. Graph: %d nodes, %d edges, %.1f MB (2-bit codes + exact edges); "
                        .formatted(c.n, c.docTitle.length, g.size(), g.edges(), g.bytes() / 1e6))
                .append("float32 embeddings of the chunks alone: %.1f MB.\n\n".formatted(c.n * 768 * 4 / 1e6))
                .append("%d single-hop queries (span of one chunk), %d two-hop queries (spans of two chunks in "
                        .formatted(singles.size(), pairs.size()))
                .append("different documents sharing a rare term).\n\n")
                .append("| Method | Full precision at query time | Index MB | 1-hop R@10 vs exact | 1-hop source@1 | ")
                .append("1-hop source@10 | 1-hop MRR | 2-hop both@10 | 2-hop both@20 | 2-hop any@10 | ms/query | ")
                .append("nodes scored/query | accepted set (k-free) |\n")
                .append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (Method m : methods) {
            log("running " + m.name());
            double rec = 0;
            double s1 = 0;
            double s10 = 0;
            double mrr = 0;
            long ns = 0;
            long scored = 0;
            long accepted = 0;
            int accN = 0;
            for (Single s : singles) {
                long t0 = System.nanoTime();
                Result r = m.run(s.q());
                ns += System.nanoTime() - t0;
                scored += r.scored();
                if (r.accepted() >= 0) {
                    accepted += r.accepted();
                    accN++;
                }
                Set<Integer> top10 = set(r.ranked(), 10);
                int hit = 0;
                for (int t : s.exactTop10()) {
                    hit += top10.contains(t) ? 1 : 0;
                }
                rec += hit / (double) s.exactTop10().length;
                int rank = indexOf(r.ranked(), s.source());
                s1 += rank == 0 ? 1 : 0;
                s10 += rank >= 0 && rank < 10 ? 1 : 0;
                mrr += rank >= 0 ? 1.0 / (rank + 1) : 0;
            }
            double both10 = 0;
            double both20 = 0;
            double any10 = 0;
            for (Pair p : pairs) {
                long t0 = System.nanoTime();
                Result r = m.run(p.q());
                ns += System.nanoTime() - t0;
                scored += r.scored();
                if (r.accepted() >= 0) {
                    accepted += r.accepted();
                    accN++;
                }
                int ra = indexOf(r.ranked(), p.a());
                int rb = indexOf(r.ranked(), p.b());
                both10 += ra >= 0 && ra < 10 && rb >= 0 && rb < 10 ? 1 : 0;
                both20 += ra >= 0 && ra < 20 && rb >= 0 && rb < 20 ? 1 : 0;
                any10 += (ra >= 0 && ra < 10) || (rb >= 0 && rb < 10) ? 1 : 0;
            }
            int ns1 = singles.size();
            int ns2 = Math.max(1, pairs.size());
            int all = ns1 + pairs.size();
            String acc = accN == 0 ? "—" : "%.0f".formatted((double) accepted / accN);
            md.append("| %s | %s | %.1f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f | %.0f | %s |\n"
                    .formatted(m.name(), m.fullPrecision() ? "yes" : "no", m.indexBytes() / 1e6, rec / ns1,
                            s1 / ns1, s10 / ns1, mrr / ns1, both10 / ns2, both20 / ns2, any10 / ns2,
                            ns / 1e6 / all, (double) scored / all, acc));
            csv.append("\"%s\",%s,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.0f,%s\n"
                    .formatted(m.name(), m.fullPrecision(), m.indexBytes() / 1e6, rec / ns1, s1 / ns1, s10 / ns1,
                            mrr / ns1, both10 / ns2, both20 / ns2, any10 / ns2, ns / 1e6 / all,
                            (double) scored / all, accN == 0 ? "" : "%.0f".formatted((double) accepted / accN)));
        }
        md.append("\nExample two-hop queries:\n\n");
        for (int i = 0; i < Math.min(3, pairs.size()); i++) {
            Pair p = pairs.get(i);
            md.append("- shared term `%s`: \"%s\" → %s | %s\n".formatted(p.term(), p.q().text(), c.label(p.a()),
                    c.label(p.b())));
        }
        Files.writeString(out.resolve("results.md"), md);
        Files.writeString(out.resolve("results.csv"), csv);
        log("wrote results/results.md and results/results.csv");
        System.out.println(md);
    }

    /** Two chunks from different documents that share a term with document frequency 2..30. */
    private static List<Pair> twoHop(Corpus c, TurboGraph g, Corpus.Embedder emb, Random rnd, int n) throws Exception {
        Map<Integer, List<Integer>> byTerm = new HashMap<>();
        for (int t = g.chunks; t < g.chunks + g.terms; t++) {
            List<Integer> cs = new ArrayList<>();
            for (int v : g.adj[t]) {
                if (g.kind[v] == TurboGraph.Kind.CHUNK && c.text[v].length() >= 300) {
                    cs.add(v);
                }
            }
            if (cs.size() >= 2 && cs.size() <= 30) {
                byTerm.put(t, cs);
            }
        }
        List<Integer> termIds = new ArrayList<>(byTerm.keySet());
        java.util.Collections.sort(termIds);
        List<Pair> out = new ArrayList<>();
        Set<Integer> usedTerms = new HashSet<>();
        int guard = 0;
        while (out.size() < n && guard++ < n * 50 && !termIds.isEmpty()) {
            int t = termIds.get(rnd.nextInt(termIds.size()));
            if (!usedTerms.add(t)) {
                continue;
            }
            List<Integer> cs = byTerm.get(t);
            int a = cs.get(rnd.nextInt(cs.size()));
            int b = -1;
            for (int tries = 0; tries < 20; tries++) {
                int cand = cs.get(rnd.nextInt(cs.size()));
                if (c.doc[cand] != c.doc[a]) {
                    b = cand;
                    break;
                }
            }
            if (b < 0) {
                continue;
            }
            String text = span(c.text[a], 90) + " ... " + span(c.text[b], 90);
            float[] e = emb.query(text);
            out.add(new Pair(new Query(text, e, g.codec.rotate(e)), a, b, g.label[t]));
        }
        return out;
    }

    static String span(String text, int len) {
        String flat = text.replaceAll("\\s+", " ").trim();
        int mid = flat.length() / 3;
        int start = Math.max(0, flat.lastIndexOf(' ', mid));
        return flat.substring(start, Math.min(flat.length(), start + len)).trim();
    }

    private static Set<Integer> set(int[] a, int k) {
        Set<Integer> s = new HashSet<>();
        for (int i = 0; i < Math.min(k, a.length); i++) {
            s.add(a[i]);
        }
        return s;
    }

    private static int indexOf(int[] a, int x) {
        for (int i = 0; i < a.length; i++) {
            if (a[i] == x) {
                return i;
            }
        }
        return -1;
    }

    private static void log(String s) {
        System.err.printf("[%tT] %s%n", System.currentTimeMillis(), s);
    }
}
