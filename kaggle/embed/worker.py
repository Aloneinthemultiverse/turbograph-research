"""Embeds everything TurboVec-HippoRAG needs, on a Kaggle GPU.

Inputs: the corpora/questions dataset and the turbograph-openie kernel output
(<ds>.facts.jsonl). Model: nomic-embed-text-v1.5 (same model Ollama serves as
nomic-embed-text), with its search_document / search_query prefixes.
Writes per dataset, float16 little-endian, 768-d rows:
  <ds>.passages.f16, <ds>.questions.f16,
  <ds>.entities.txt + <ds>.entities.f16   (unique normalised entity phrases)
  <ds>.facts.tsv    + <ds>.facts.f16      (passage idx, subject, relation, object)
"""
import glob
import json
import os
import subprocess
import time

WORK = "/kaggle/working"
ORDER = ["hotpotqa", "2wikimultihopqa", "musique"]


def norm(s):
    return " ".join(str(s).lower().split())


def main():
    subprocess.run("pip install -q sentence-transformers einops", shell=True, check=True)
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cuda")
    model.half()

    def emb(texts, prefix, path):
        t0 = time.time()
        v = model.encode([prefix + t for t in texts], batch_size=256, normalize_embeddings=False,
                         convert_to_numpy=True, show_progress_bar=False)
        v.astype("<f2").tofile(path)
        print(f"  {os.path.basename(path)}: {len(texts)} in {time.time() - t0:.0f}s", flush=True)

    inputs = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*", recursive=True) if os.path.isfile(p)}
    for ds in ORDER:
        facts_file = inputs.get(f"{ds}.facts.jsonl")
        if facts_file is None:
            print("missing facts for", ds, flush=True)
            continue
        rows = [l.rstrip("\n").split("\t") for l in open(inputs[f"{ds}.corpus.tsv"], encoding="utf-8")]
        emb([r[1] + ": " + r[2] for r in rows], "search_document: ", f"{WORK}/{ds}.passages.f16")
        qs = [l.rstrip("\n").split("\t") for l in open(inputs[f"{ds}.questions.tsv"], encoding="utf-8")]
        emb([q[1] for q in qs], "search_query: ", f"{WORK}/{ds}.questions.f16")

        ents = {}
        facts = []
        for line in open(facts_file, encoding="utf-8"):
            d = json.loads(line)
            for e in d["entities"]:
                ents.setdefault(norm(e), None)
            for s, r, o in d["triples"]:
                ents.setdefault(norm(s), None)
                ents.setdefault(norm(o), None)
                facts.append((d["idx"], norm(s), norm(r), norm(o)))
        names = [e for e in ents if e]
        with open(f"{WORK}/{ds}.entities.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
        emb(names, "search_document: ", f"{WORK}/{ds}.entities.f16")
        with open(f"{WORK}/{ds}.facts.tsv", "w", encoding="utf-8") as f:
            for idx, s, r, o in facts:
                f.write(f"{idx}\t{s}\t{r}\t{o}\n")
        emb([f"{s} {r} {o}" for _, s, r, o in facts], "search_document: ", f"{WORK}/{ds}.facts.f16")
        print(f"{ds}: {len(rows)} passages, {len(names)} entities, {len(facts)} facts", flush=True)


if __name__ == "__main__":
    main()
