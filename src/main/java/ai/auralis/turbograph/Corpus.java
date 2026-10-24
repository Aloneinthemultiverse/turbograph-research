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

    final int n;
    final String[] text;
    final int[] doc;
    final String[] docTitle;
    final int[] ordinal;
    final float[][] emb;

    private Corpus(int n, String[] text, int[] doc, String[] docTitle, int[] ordinal, float[][] emb) {
        this.n = n;
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
