from pathlib import Path
from datasets import load_dataset


def iter_custom_text(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"custom corpus file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_wikipedia_hf(candidates, token=None, max_lines=None):
    last_error = None
    for cand in candidates:
        ds_name = cand["dataset"]
        cfg = cand.get("config")
        split = cand.get("split", "train")
        text_col = cand.get("text_column", "text")
        try:
            print(f"[CORPUS] Try HF dataset={ds_name}, config={cfg}, split={split}")
            ds = load_dataset(ds_name, cfg, split=split, token=token) if cfg else load_dataset(ds_name, split=split, token=token)
            count = 0
            for row in ds:
                text = row.get(text_col, "") if isinstance(row, dict) else ""
                if text:
                    yield text
                    count += 1
                    if max_lines is not None and count >= max_lines:
                        return
            return
        except Exception as e:
            last_error = e
            print(f"[WARN] corpus candidate failed: {repr(e)}")
    raise RuntimeError(f"All Wikipedia corpus candidates failed. Last error: {repr(last_error)}")


def build_corpus_line_iterator(cfg, token=None):
    source = cfg.get("source", "benchmark_only")
    max_lines = cfg.get("max_corpus_lines")
    if source == "custom_text":
        return iter_custom_text(cfg["custom_text_path"])
    if source == "wikipedia_hf":
        return iter_wikipedia_hf(cfg.get("wikipedia_candidates", []), token=token, max_lines=max_lines)
    if source == "benchmark_only":
        return iter([])
    raise ValueError(f"Unsupported corpus source: {source}")
