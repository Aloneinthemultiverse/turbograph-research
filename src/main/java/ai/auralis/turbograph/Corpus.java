package ai.auralis.turbograph;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The Auralis knowledge base as plain arrays: chunk text, document, and the full
 * float embedding (kept only for ground truth and exact re-scoring). Query
 * embeddings come from the same Ollama model and prefix Auralis uses, cached on disk.
 */
public final class Corpus {

    static final Pattern TERM = Pattern.compile(
            "\\b(?:[A-Z][A-Z0-9]{1,11}(?:[-_][A-Z0-9]{1,8})?|[A-Za-z]{2,10}\\d{1,5}[A-Za-z0-9]{0,6})\\b");
    static final Set<String> STOP = Set.of(
            "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "ARE", "NOT", "BUT", "ALL", "ANY",
            "PDF", "FIG", "TABLE", "NOTE", "SEE", "PAGE", "ET", "AL", "II", "III", "IV", "VI", "OF",
            "IN", "ON", "TO", "IS", "IT", "BY", "OR", "AS", "AN", "AT", "BE", "WE", "IF", "NO", "DO");

    /** Matches capitalised names (Chris Evans, Warsaw Pact) for Wikipedia-style corpora. */
    static final Pattern NAME = Pattern.compile(
            "\\b\\p{Lu}[\\p{L}\\p{N}'\u2019.-]*(?:\\s+(?:of|the|de|del|da|von|van|and|for|on|in)?\\s*\\p{Lu}[\\p{L}\\p{N}'\u2019.-]*){0,4}");
    static final Set<String> NAME_STOP = Set.of(
            "THE", "A", "AN", "IN", "ON", "AT", "HE", "SHE", "IT", "THEY", "HIS", "HER", "ITS", "THIS", "THAT",
            "THESE", "THOSE", "AS", "BY", "FOR", "FROM", "AFTER", "BEFORE", "DURING", "WHEN", "WHERE", "WHO",
            "WHAT", "WHICH", "HOW", "WHY", "IS", "WAS", "ARE", "WERE", "AND", "BUT", "OR", "OF", "TO", "WITH",
            "IF", "ALSO", "BOTH", "THERE", "THEIR", "WE", "I", "YOU", "ONE", "SOME", "MANY", "MOST", "OTHER");

    final int n;
    /** True for Wikipedia-style corpora: entities are capitalised names and passage titles. */
    final boolean wiki;
    final String[] text;
    final int[] doc;
    final String[] docTitle;
    final int[] ordinal;
    final float[][] emb;

    private Corpus(int n, String[] text, int[] doc, String[] docTitle, int[] ordinal, float[][] emb) {
        this(n, false, text, doc, docTitle, ordinal, emb);
    }

    private Corpus(int n, boolean wiki, String[] text, int[] doc, String[] docTitle, int[] ordinal, float[][] emb) {
        this.n = n;
        this.wiki = wiki;
        this.text = text;
        this.doc = doc;
        this.docTitle = docTitle;
        this.ordinal = ordinal;
        this.emb = emb;
    }

    static Corpus load(String url, String user, String pass) throws Exception {
        List<String> texts = new ArrayList<>();
        List<Integer> docs = new ArrayList<>();
        List<Integer> ords = new ArrayList<>();
        List<float[]> embs = new ArrayList<>();
        Map<String, Integer> docIdx = new HashMap<>();
        List<String> titles = new ArrayList<>();
        try (Connection c = DriverManager.getConnection(url, user, pass);
             PreparedStatement ps = c.prepareStatement("""
                     SELECT c.document_id::text, d.title, c.ordinal, c.content, c.embedding::text
                     FROM chunk c JOIN source_document d ON d.id = c.document_id
                     ORDER BY c.document_id, c.ordinal""")) {
            ps.setFetchSize(2000);
            c.setAutoCommit(false);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    String d = rs.getString(1);
                    Integer di = docIdx.get(d);
                    if (di == null) {
                        di = titles.size();
                        docIdx.put(d, di);
                        titles.add(rs.getString(2));
                    }
                    docs.add(di);
                    ords.add(rs.getInt(3));
                    texts.add(rs.getString(4));
                    embs.add(parse(rs.getString(5)));
                }
            }
        }
        int n = texts.size();
        int[] doc = new int[n];
        int[] ord = new int[n];
        for (int i = 0; i < n; i++) {
            doc[i] = docs.get(i);
            ord[i] = ords.get(i);
        }
        return new Corpus(n, texts.toArray(String[]::new), doc, titles.toArray(String[]::new), ord,
                embs.toArray(float[][]::new));
    }

    /**
     * A HippoRAG benchmark corpus from scripts/prep_hipporag.py output (idx, title, text, key).
     * Passages are embedded as "title: text" with the document prefix, cached on disk.
     * Passages sharing a title form one document.
     */
    static Corpus loadTsv(Path tsv, Embedder embedder, Path embCache, Map<String, Integer> keyToIdx) throws Exception {
        List<String> lines = Files.readAllLines(tsv);
        int n = lines.size();
        String[] text = new String[n];
        int[] doc = new int[n];
        int[] ord = new int[n];
        Map<String, Integer> docIdx = new HashMap<>();
        Map<Integer, Integer> perDoc = new HashMap<>();
        List<String> titles = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            String[] f = lines.get(i).split("\t", -1);
            text[i] = f[1] + ": " + f[2];
            Integer d = docIdx.get(f[1]);
            if (d == null) {
                d = titles.size();
                docIdx.put(f[1], d);
                titles.add(f[1]);
            }
            doc[i] = d;
            ord[i] = perDoc.merge(d, 1, Integer::sum) - 1;
            keyToIdx.put(f[3], i);
        }
        float[][] emb = embedder.documents(java.util.Arrays.asList(text), embCache);
        return new Corpus(n, true, text, doc, titles.toArray(String[]::new), ord, emb);
    }

    /** Entity strings of a chunk: identifiers, plus names and the own title for wiki corpora. */
    Set<String> termsOf(int i) {
        if (!wiki) {
            return terms(text[i]);
        }
        Set<String> out = names(text[i]);
        out.add(normTitle(docTitle[doc[i]]));
        return out;
    }

    /** Entity strings named in a query, by the same extractor as the corpus. */
    Set<String> queryEntities(String q) {
        return wiki ? names(q) : terms(q);
    }

    static Set<String> names(String text) {
        Set<String> out = terms(text);
        Matcher m = NAME.matcher(text);
        while (m.find()) {
            String t = m.group().replaceAll("[.'\u2019-]+$", "").toUpperCase();
            if (!NAME_STOP.contains(t) && t.length() > 1) {
                out.add(t);
            }
        }
        return out;
    }

    /** "Chris Evans (actor)" -> "CHRIS EVANS". */
    static String normTitle(String title) {
        return title.replaceAll("\\s*\\(.*?\\)\\s*", " ").trim().toUpperCase();
    }

    String label(int i) {
        return docTitle[doc[i]] + " #" + ordinal[i];
    }

    static Set<String> terms(String text) {
        Set<String> out = new HashSet<>();
        Matcher m = TERM.matcher(text);
        while (m.find()) {
            String t = m.group().toUpperCase();
            if (!STOP.contains(t) && !t.chars().allMatch(Character::isDigit)) {
                out.add(t);
            }
        }
        return out;
    }

    static float[] parse(String literal) {
        String[] parts = literal.substring(1, literal.length() - 1).split(",");
        float[] v = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            v[i] = Float.parseFloat(parts[i]);
        }
        return v;
    }

    static double cos(float[] a, float[] b) {
        double s = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            s += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return s / Math.sqrt(na * nb + 1e-12);
    }

    /** Ollama nomic-embed-text with Auralis's query prefix; cached by text in a TSV file. */
    static final class Embedder {
        private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
        private final String base;
        private final Path cacheFile;
        private final Map<String, float[]> cache = new HashMap<>();

        Embedder(String base, Path cacheFile) throws Exception {
            this.base = base;
            this.cacheFile = cacheFile;
            if (Files.exists(cacheFile)) {
                for (String line : Files.readAllLines(cacheFile)) {
                    int tab = line.indexOf('\t');
                    if (tab > 0) {
                        cache.put(line.substring(0, tab), parse(line.substring(tab + 1)));
                    }
                }
            }
        }

        float[] query(String q) throws Exception {
            String key = q.replace('\t', ' ').replace('\n', ' ');
            float[] hit = cache.get(key);
            if (hit != null) {
                return hit;
            }
            String body = "{\"model\":\"nomic-embed-text\",\"input\":[" + json("search_query: " + q) + "]}";
            HttpResponse<String> r = http.send(HttpRequest.newBuilder(URI.create(base + "/api/embed"))
                            .timeout(Duration.ofMinutes(2)).header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                    HttpResponse.BodyHandlers.ofString());
            String s = r.body();
            int a = s.indexOf("[[");
            int b = s.indexOf("]]", a);
            if (a < 0 || b < 0) {
                throw new IllegalStateException("Ollama embed failed: " + s);
            }
            float[] v = parse(s.substring(a + 1, b + 1));
            cache.put(key, v);
            StringBuilder sb = new StringBuilder(key).append('\t').append('[');
            for (int i = 0; i < v.length; i++) {
                sb.append(i == 0 ? "" : ",").append(v[i]);
            }
            Files.writeString(cacheFile, sb.append("]\n"), java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.APPEND);
            return v;
        }

        /** Document embeddings ("search_document: " prefix), batched, cached as raw float32. */
        float[][] documents(List<String> texts, Path cache) throws Exception {
            int n = texts.size();
            int dim = 768;
            if (Files.exists(cache) && Files.size(cache) == (long) n * dim * 4) {
                java.nio.FloatBuffer fb = java.nio.ByteBuffer.wrap(Files.readAllBytes(cache))
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                float[][] out = new float[n][dim];
                for (float[] row : out) {
                    fb.get(row);
                }
                return out;
            }
            float[][] out = new float[n][];
            int batch = 32;
            long t0 = System.currentTimeMillis();
            for (int i = 0; i < n; i += batch) {
                StringBuilder body = new StringBuilder("{\"model\":\"nomic-embed-text\",\"input\":[");
                for (int j = i; j < Math.min(n, i + batch); j++) {
                    body.append(j == i ? "" : ",").append(json("search_document: " + texts.get(j)));
                }
                body.append("]}");
                HttpResponse<String> r = http.send(HttpRequest.newBuilder(URI.create(base + "/api/embed"))
                                .timeout(Duration.ofMinutes(5)).header("Content-Type", "application/json")
                                .POST(HttpRequest.BodyPublishers.ofString(body.toString())).build(),
                        HttpResponse.BodyHandlers.ofString());
                String s = r.body();
                int a = s.indexOf("[[");
                int b = s.indexOf("]]", a);
                if (a < 0 || b < 0) {
                    throw new IllegalStateException("Ollama embed failed: " + s);
                }
                String[] rows = s.substring(a + 2, b).split("\\],\\[");
                for (int j = 0; j < rows.length; j++) {
                    out[i + j] = parse("[" + rows[j] + "]");
                }
                if ((i / batch) % 50 == 0) {
                    System.err.printf("  embedded %d/%d (%.0f s)%n", i + rows.length, n,
                            (System.currentTimeMillis() - t0) / 1000.0);
                }
            }
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(n * dim * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (float[] row : out) {
                for (float x : row) {
                    bb.putFloat(x);
                }
            }
            Files.write(cache, bb.array());
            return out;
        }

        private static String json(String s) {
            StringBuilder sb = new StringBuilder("\"");
            for (char ch : s.toCharArray()) {
                switch (ch) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n', '\r', '\t' -> sb.append(' ');
                    default -> {
                        if (ch < 0x20) {
                            sb.append(' ');
                        } else {
                            sb.append(ch);
                        }
                    }
                }
            }
            return sb.append('"').toString();
        }
    }
}
