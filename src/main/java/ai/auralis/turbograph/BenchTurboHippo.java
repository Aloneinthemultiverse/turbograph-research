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
import ai.auralis.turbograph.TurboHippo.Config;

/**
 * TurboVec-HippoRAG vs HippoRAG 2 (float) on the HippoRAG benchmarks, with one shared
 * fact graph and one shared embedding model (nomic-embed-text-v1.5, computed on Kaggle).
 *
 * <pre>java ... BenchTurboHippo hotpotqa</pre>
 */
public final class BenchTurboHippo {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        String ds = args.length > 0 ? args[0] : "hotpotqa";
        Path data = Path.of("data/hipporag");
        Path th = data.resolve("turbohippo");
        Map<String, Integer> keyToIdx = new HashMap<>();
        Corpus c = Corpus.loadTsv(data.resolve(ds + ".corpus.tsv"), keyToIdx,
                texts -> Corpus.readF16(th.resolve(ds + ".passages.f16"), 768));
        log("%s: %d passages".formatted(ds, c.n));
        TurboHippo g = TurboHippo.build(th, ds, c.emb);
        log("TurboVec-HippoRAG: %d nodes (%d passages, %d entities, %d facts), %d exact edges; "
                .formatted(g.size(), g.P, g.E, g.F, g.edges())
                + "2-bit %.1f MB vs float %.1f MB; built in %d ms".formatted(g.bytes2bit() / 1e6,
                g.bytesFloat() / 1e6, g.buildMs));

        float[][] qEmb = Corpus.readF16(th.resolve(ds + ".questions.f16"), 768);
        List<String> qLines = Files.readAllLines(data.resolve(ds + ".questions.tsv"));
        List<Query> qs = new ArrayList<>();
        List<int[]> gold = new ArrayList<>();
        for (int i = 0; i < qLines.size() && i < qEmb.length; i++) {
            String[] f = qLines.get(i).split("\t", -1);
            List<Integer> sup = new ArrayList<>();
            for (String k : f[2].split("\\|")) {
                Integer x = keyToIdx.get(k);
                if (x != null && !sup.contains(x)) {
                    sup.add(x);
                }
            }
            if (!sup.isEmpty()) {
                qs.add(new Query(f[1], qEmb[i], g.codec.rotate(qEmb[i])));
                gold.add(sup.stream().mapToInt(Integer::intValue).toArray());
            }
        }
        log(qs.size() + " questions");

        Method dense2bit = new Method() {
            public String name() { return "Dense 2-bit (TurboVec flat)"; }
            public boolean fullPrecision() { return false; }
            public long indexBytes() { return (long) g.P * (g.codec.paddedDim() / 4 + 4); }
            public Result run(Query q) {
                float[] s = new float[g.P];
                for (int i = 0; i < g.P; i++) {
                    s[i] = g.score(q, i, true);
                }
                return new Result(Methods.topK(s, 100), g.P, -1);
            }
        };
        Method bm25 = Methods.bm25(c);
        Method turbo = g.method(new Config("TurboVec-HippoRAG (2-bit, local push, HippoRAG edges)",
                true, true, false, false, 0));
        Method turboFull = g.method(new Config("TurboVec-HippoRAG + fact nodes + SIMILAR edges (2-bit, local push)",
                true, true, true, true, 0));
        Method turboFused = g.method(new Config("TurboVec-HippoRAG fused (2-bit, local push, lambda=0.1)",
                true, true, false, false, 0.1));
        List<Method> methods = List.of(
                Methods.exact(c),
                dense2bit,
                bm25,
                g.method(new Config("HippoRAG 2 (float vectors, global PPR)", false, false, false, false, 0)),
                g.method(new Config("HippoRAG 2 graph + 2-bit vectors (global PPR)", true, false, false, false, 0)),
                turbo, turboFull, turboFused,
                Methods.rrf("Hybrid RRF: dense 2-bit + BM25 + TurboVec-HippoRAG", List.of(dense2bit, bm25, turbo)));

        StringBuilder md = new StringBuilder("## %s — TurboVec-HippoRAG, %d questions, %d passages\n\n"
                .formatted(ds, qs.size(), c.n))
                .append("Graph: %d nodes (%d passages, %d entities, %d facts), %d exact edges. "
                        .formatted(g.size(), g.P, g.E, g.F, g.edges()))
                .append("Index: 2-bit %.1f MB vs float32 %.1f MB (%.1fx smaller).\n\n"
                        .formatted(g.bytes2bit() / 1e6, g.bytesFloat() / 1e6, (double) g.bytesFloat() / g.bytes2bit()))
                .append("| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q |\n")
                .append("|---|---|---|---|---|---|---|---|\n");
        for (Method m : methods) {
            log("running " + m.name());
            double r2 = 0;
            double r5 = 0;
            double all5 = 0;
            long ns = 0;
            long scored = 0;
            for (int i = 0; i < qs.size(); i++) {
                long t0 = System.nanoTime();
                Result r = m.run(qs.get(i));
                ns += System.nanoTime() - t0;
                scored += r.scored();
                int in2 = 0;
                int in5 = 0;
                for (int s : gold.get(i)) {
                    int rank = indexOf(r.ranked(), s);
                    in2 += rank >= 0 && rank < 2 ? 1 : 0;
                    in5 += rank >= 0 && rank < 5 ? 1 : 0;
                }
                r2 += (double) in2 / gold.get(i).length;
                r5 += (double) in5 / gold.get(i).length;
                all5 += in5 == gold.get(i).length ? 1 : 0;
            }
            int n = qs.size();
            md.append("| %s | %s | %.1f | %.3f | %.3f | %.3f | %.1f | %.0f |\n".formatted(m.name(),
                    m.fullPrecision() ? "yes" : "no", m.indexBytes() / 1e6, r2 / n, r5 / n, all5 / n,
                    ns / 1e6 / n, (double) scored / n));
        }
        Files.writeString(Path.of("results/turbohippo-" + ds + ".md"), md);
        log("wrote results/turbohippo-" + ds + ".md");
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
