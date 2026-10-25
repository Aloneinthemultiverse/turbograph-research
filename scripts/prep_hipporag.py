"""Converts the HippoRAG benchmark JSON (HotpotQA, MuSiQue, 2WikiMultiHopQA) to TSV.

    python -I scripts/prep_hipporag.py data/hipporag <dataset>

Writes data/hipporag/<dataset>.corpus.tsv (idx, title, text, key) and
data/hipporag/<dataset>.questions.tsv (id, question, supporting passage keys joined by |).
Supporting passages are matched by title + start of text (MuSiQue repeats titles).
"""
import json
import os
import sys


def clean(s):
    return " ".join(str(s).split())


def key(title, text):
    """Title plus the first 40 letters/digits of the text: unique even when titles repeat."""
    t = "".join(ch for ch in str(text).lower() if ch.isalnum())[:40]
    return clean(title) + "::" + t


def main(folder, name):
    corpus = json.load(open(os.path.join(folder, name + "_corpus.json"), encoding="utf-8"))
    qs = json.load(open(os.path.join(folder, name + ".json"), encoding="utf-8"))
    with open(os.path.join(folder, name + ".corpus.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for i, p in enumerate(corpus):
            f.write(f"{i}\t{clean(p['title'])}\t{clean(p['text'])}\t{key(p['title'], p['text'])}\n")
    n = 0
    with open(os.path.join(folder, name + ".questions.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for q in qs:
            if "supporting_facts" in q:
                ctx = {t: " ".join(sents) for t, sents in q["context"]}
                titles = []
                for t, _ in q["supporting_facts"]:
                    k = key(t, ctx.get(t, ""))
                    if k not in titles:
                        titles.append(k)
            else:
                titles = [key(p["title"], p["paragraph_text"]) for p in q["paragraphs"] if p.get("is_supporting")]
            qid = q.get("_id", q.get("id"))
            f.write(f"{qid}\t{clean(q['question'])}\t{'|'.join(clean(t) for t in titles)}\n")
            n += 1
    print(name, "corpus", len(corpus), "questions", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
