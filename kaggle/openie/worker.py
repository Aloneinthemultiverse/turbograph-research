"""OpenIE fact extraction on a Kaggle GPU (2x T4), for TurboVec-HippoRAG.

For every passage of the HippoRAG benchmark corpora (attached as a private dataset
of *.corpus.tsv files), an instruction model extracts named entities and
(subject, relation, object) triples, as in HippoRAG's offline indexing.
Writes /kaggle/working/<dataset>.facts.jsonl: {"idx", "entities", "triples"}.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
WORK = "/kaggle/working"
ORDER = ["hotpotqa", "2wikimultihopqa", "musique"]

PROMPT = """Extract a knowledge graph from the passage.

1. List the named entities (people, places, organisations, works, events, dates, numbers that matter).
2. List factual triples [subject, relation, object]. Each triple must contain at least one of the
   named entities; use the full entity name (resolve pronouns such as "he" or "the film");
   keep relations short (e.g. "starred in", "born in", "capital of", "directed").

Answer with JSON only: {{"entities": [...], "triples": [[subject, relation, object], ...]}}

Passage title: {title}
Passage: {text}"""


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def parse(out):
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return [], []
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [], []
    ents = [str(e).strip() for e in d.get("entities", []) if str(e).strip()]
    triples = []
    for t in d.get("triples", []):
        if isinstance(t, list) and len(t) == 3 and all(str(x).strip() for x in t):
            triples.append([str(x).strip() for x in t])
    return ents, triples


def main():
    sh("pip install -q vllm")
    # vLLM brings its own torch; Kaggle's preinstalled torchaudio then mismatches CUDA and breaks imports.
    sh("pip uninstall -y -q torchaudio")
    from vllm import LLM, SamplingParams

    files = {os.path.basename(p).split(".")[0]: p for p in glob.glob("/kaggle/input/**/*.corpus.tsv", recursive=True)}
    print("corpora:", files, flush=True)
    llm = LLM(model=MODEL, dtype="half", tensor_parallel_size=2, max_model_len=4096,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    params = SamplingParams(temperature=0.0, max_tokens=700)
    for ds in ORDER:
        if ds not in files:
            continue
        rows = [line.rstrip("\n").split("\t") for line in open(files[ds], encoding="utf-8")]
        msgs = [[{"role": "user", "content": PROMPT.format(title=r[1], text=r[2][:6000])}] for r in rows]
        t0 = time.time()
        outs = llm.chat(msgs, params, use_tqdm=True)
        n_t = 0
        with open(f"{WORK}/{ds}.facts.jsonl", "w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                ents, triples = parse(o.outputs[0].text)
                n_t += len(triples)
                f.write(json.dumps({"idx": int(r[0]), "entities": ents, "triples": triples},
                                   ensure_ascii=False) + "\n")
        print(f"{ds}: {len(rows)} passages, {n_t} triples, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
