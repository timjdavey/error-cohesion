"""Download and parse the Hot Mess probability runs (huggingface.co/datasets/hot-mess/hot-mess-data).

Each run of a config is one `samples_*.jsonl` file, with one row per question
and filter. The `match-prob` rows hold the answer as four probabilities, like
`<PROB>0.70, 0.05, 0.25, 0.00</PROB>`. Grouping one question's rows across the
runs gives a (runs x 4) array, the same shape the toy uses.

Parsing follows `load_model_data` in the paper's `scripts/utils.py`
(github.com/haeggee/hot-mess-of-ai), so our numbers line up with theirs.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = "hot-mess/hot-mess-data"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hot-mess"  # gitignored
CACHE = ROOT / "data" / "cache"

PROB_TAG = re.compile(r"<PROB>(.*?)</PROB>", re.DOTALL)
PROB_VALUE = re.compile(r"\b(?:0\.\d+|1\.0*|0\.0*)\b")
ANY_NUMBER = re.compile(r"\b\d+\.?\d*\b")
TWO_DP = re.compile(r"\b\d+\.\d{2}\b")
LETTER = re.compile(r"\(([ABCD])\)")


@dataclass
class Question:
    """One question's samples across every run of a config."""

    doc_id: int
    probs: np.ndarray  # runs x 4, each row sums to 1
    gold: int  # index of the correct option
    parse: np.ndarray  # how each row was read: "tag", "floats", "letter" or "none"


def download(config):
    """Fetch only the samples files for `config`, a folder under `mcq/`, and return its path."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=DATA,
        allow_patterns=[f"mcq/{config}/*/samples_*.jsonl"],
        max_workers=8,
    )
    return DATA / "mcq" / config


def parse_probs(resp):
    """Read four probabilities from a response, the way the paper's loader does.

    Returns the probabilities and how they were found. A response with no
    usable answer becomes uniform, as in the paper.
    """
    probs, how = None, "none"
    tag = PROB_TAG.search(resp)
    if tag:
        values = PROB_VALUE.findall(tag.group(1)) or []
        if len(values) < 4:
            values = ANY_NUMBER.findall(tag.group(1))
        if len(values) >= 4:
            probs, how = np.array([float(v) for v in values[:4]]), "tag"
    else:
        floats = [float(v) for v in TWO_DP.findall(resp)[-4:]]
        if len(floats) == 4 and abs(sum(floats) - 1) < 0.01:
            probs, how = np.array(floats), "floats"
        elif letters := LETTER.findall(resp):
            probs, how = np.eye(4)[ord(letters[-1]) - ord("A")], "letter"
    if probs is None or probs.sum() <= 0:
        return np.full(4, 0.25), "none"
    return probs / probs.sum(), how


def gold_index(target):
    """Return the correct option from a target like `<PROB>0.00, 1.00, 0.00, 0.00</PROB>`."""
    return int(np.argmax([float(v) for v in PROB_TAG.search(target).group(1).split(",")]))


def read_runs(folder):
    """Parse every samples file under `folder` into {model: {doc_id: Question}}."""
    out = {}
    for model_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        rows = {}
        for path in sorted(model_dir.glob("samples_*.jsonl")):
            with path.open() as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("filter") != "match-prob":
                        continue
                    probs, how = parse_probs(rec["resps"][0][0])
                    gold = gold_index(rec["target"])
                    r = rows.setdefault(rec["doc_id"], {"gold": gold, "p": [], "how": []})
                    r["p"].append(probs)
                    r["how"].append(how)
        out[model_dir.name] = {
            doc_id: Question(doc_id, np.array(r["p"]), r["gold"], np.array(r["how"]))
            for doc_id, r in sorted(rows.items())
        }
    return out


def load(config):
    """Return {doc_id: Question} for `config`, downloading and caching as needed."""
    cache = CACHE / f"{config}.npz"
    if cache.exists():
        runs = unpack(np.load(cache, allow_pickle=False))
    else:
        runs = read_runs(download(config))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **pack(runs))
    (questions, *rest) = runs.values()
    assert not rest, f"{config} has more than one model folder"
    return questions


def pack(runs):
    """Flatten runs into arrays that `np.savez` can store without pickling."""
    out = {}
    for model, questions in runs.items():
        for q in questions.values():
            key = f"{model}|{q.doc_id}"
            out[f"{key}|probs"] = q.probs
            out[f"{key}|gold"] = np.array(q.gold)
            out[f"{key}|parse"] = q.parse
    return out


def unpack(npz):
    """Rebuild runs from the arrays `pack` wrote."""
    runs = {}
    for name in npz.files:
        model, doc_id, field = name.split("|")
        if field != "probs":
            continue
        key = f"{model}|{doc_id}"
        runs.setdefault(model, {})[int(doc_id)] = Question(
            int(doc_id), npz[name], int(npz[f"{key}|gold"]), npz[f"{key}|parse"]
        )
    return {m: dict(sorted(qs.items())) for m, qs in runs.items()}
