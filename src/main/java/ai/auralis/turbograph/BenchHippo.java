package ai.auralis.turbograph;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import ai.auralis.turbograph.Methods.Method;
import ai.auralis.turbograph.Methods.Query;
import ai.auralis.turbograph.Methods.Result;

/**
 * Multi-hop retrieval on the HippoRAG benchmark files (HotpotQA, MuSiQue,
 * 2WikiMultiHopQA; 1000 questions each, as in the HippoRAG papers). Reports
 * passage Recall@2 / Recall@5 of the supporting passages (HippoRAG's metric),
 * All@5 (every supporting passage in the top 5), latency, nodes scored and index size.
 *
 * <pre>java ... BenchHippo hotpotqa [questions]</pre>
 */
public final class BenchHippo {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        String ds = args.length > 0 ? args[0] : "hotpotqa";
        int limit = args.length > 1 ? Integer.parseInt(args[1]) : 1000;
        Path data = Path.of("data/hipporag");
        Path out = Path.of("results");
        Files.createDirectories(out.resolve("cache"));
        Corpus.Embedder emb = new Corpus.Embedder("http://localhost:11434", out.resolve("cache/queries-" + ds + ".tsv"));
        Map<String, Integer> keyToIdx = new HashMap<>();
        log("loading and embedding " + ds + " corpus");
        Corpus c = Corpus.loadTsv(data.resolve(ds + ".corpus.tsv"), emb, out.resolve("cache/" + ds + ".emb"), keyToIdx);
        TurboGraph g = TurboGraph.build(c, 12, 24, 48);
        log("%s: %d passages; graph %d nodes (%d entities), %d edges, %.1f MB, built in %d ms"
                .formatted(ds, c.n, g.size(), g.terms, g.edges(), g.bytes() / 1e6, g.buildMs));

        List<Query> qs = new ArrayList<>();
        List<int[]> gold = new ArrayList<>();
        for (String line : Files.readAllLines(data.resolve(ds + ".questions.tsv"))) {
            if (qs.size() >= limit) {
                break;
            }
            String[] f = line.split("\t", -1);
            List<Integer> sup = new ArrayList<>();
            for (String k : f[2].split("\\|")) {
                Integer i = keyToIdx.get(k);
                if (i != null && !sup.contains(i)) {
                    sup.add(i);
                }
            }
            if (sup.isEmpty()) {
                continue;
            }
            float[] e = emb.query(f[1]);
            qs.add(new Query(f[1], e, g.codec.rotate(e)));
            gold.add(sup.stream().mapToInt(Integer::intValue).toArray());
        }
        log("%d questions".formatted(qs.size()));

        Method flat = Methods.flat2bit(g);
        Method flatRescore = Methods.flat2bitRescore(c, g, 100);
        Method bm25 = Methods.bm25(c);
        Method pprScan = Methods.pushPpr(g, Methods.Seeds.SCAN, 0.15, 1e-4, 4);
        Method fusedLocal = Methods.pushPpr(g, Methods.Seeds.LOCAL, 0.15, 1e-4, 4, 0.1);
        Method fusedScan = Methods.pushPpr(g, Methods.Seeds.SCAN, 0.15, 1e-4, 4, 0.1);
        List<Method> methods = new ArrayList<>(List.of(
                Methods.exact(c), flat, flatRescore, bm25,
                Methods.kHop(g), Methods.beamWalk(g, 200), Methods.thresholdExpansion(g, 0.08f),
                Methods.pushPpr(g, Methods.Seeds.LOCAL, 0.15, 1e-4, 4), pprScan,
                Methods.pushPpr(g, Methods.Seeds.LOCAL, 0.15, 1e-4, 4, 0.05), fusedLocal,
                Methods.pushPpr(g, Methods.Seeds.LOCAL, 0.15, 1e-4, 4, 0.2),
                Methods.pushPpr(g, Methods.Seeds.SCAN, 0.15, 1e-4, 4, 0.05), fusedScan,
                Methods.pushPpr(g, Methods.Seeds.SCAN, 0.15, 1e-4, 4, 0.2),
                Methods.rescored(fusedScan, c),
                Methods.rrf("Hybrid RRF: flat 2-bit + BM25", List.of(flat, bm25)),
                Methods.rrf("Hybrid RRF: flat 2-bit + BM25 + fused PPR SCAN", List.of(flat, bm25, fusedScan)),
                Methods.rrf("Hybrid RRF: flat+rescore + BM25 + fused PPR SCAN", List.of(flatRescore, bm25, fusedScan))));

        StringBuilder md = new StringBuilder("## %s — %d questions, %d passages\n\n".formatted(ds, qs.size(), c.n))
                .append("Graph: %d nodes (%d passages, %d entities, %d documents), %d exact edges, %.1f MB; "
                        .formatted(g.size(), g.chunks, g.terms, g.docs, g.edges(), g.bytes() / 1e6))
                .append("float32 passage embeddings: %.1f MB.\n\n".formatted(c.n * 768 * 4 / 1e6))
                .append("| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q | accepted (k-free) |\n")
                .append("|---|---|---|---|---|---|---|---|---|\n");
        StringBuilder csv = new StringBuilder("dataset,method,full_precision,index_mb,r2,r5,all5,ms,nodes,accepted\n");
        for (Method m : methods) {
            log("running " + m.name());
            double r2 = 0;
            double r5 = 0;
            double all5 = 0;
            long ns = 0;
            long scored = 0;
            long acc = 0;
            int accN = 0;
            for (int i = 0; i < qs.size(); i++) {
                long t0 = System.nanoTime();
                Result r = m.run(qs.get(i));
                ns += System.nanoTime() - t0;
                scored += r.scored();
                if (r.accepted() >= 0) {
                    acc += r.accepted();
                    accN++;
                }
                int[] sup = gold.get(i);
                int in2 = 0;
                int in5 = 0;
                for (int s : sup) {
                    int rank = indexOf(r.ranked(), s);
                    in2 += rank >= 0 && rank < 2 ? 1 : 0;
                    in5 += rank >= 0 && rank < 5 ? 1 : 0;
                }
                r2 += (double) in2 / sup.length;
                r5 += (double) in5 / sup.length;
                all5 += in5 == sup.length ? 1 : 0;
            }
            int n = qs.size();
            String a = accN == 0 ? "—" : "%.0f".formatted((double) acc / accN);
            md.append("| %s | %s | %.1f | %.3f | %.3f | %.3f | %.1f | %.0f | %s |\n".formatted(m.name(),
                    m.fullPrecision() ? "yes" : "no", m.indexBytes() / 1e6, r2 / n, r5 / n, all5 / n,
                    ns / 1e6 / n, (double) scored / n, a));
            csv.append("%s,\"%s\",%s,%.2f,%.4f,%.4f,%.4f,%.2f,%.0f,%s\n".formatted(ds, m.name(), m.fullPrecision(),
                    m.indexBytes() / 1e6, r2 / n, r5 / n, all5 / n, ns / 1e6 / n, (double) scored / n,
                    accN == 0 ? "" : "%.0f".formatted((double) acc / accN)));
        }
        Files.writeString(out.resolve(ds + ".md"), md);
        Files.writeString(out.resolve(ds + ".csv"), csv);
        log("wrote results/" + ds + ".md");
        System.out.println(md);
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
