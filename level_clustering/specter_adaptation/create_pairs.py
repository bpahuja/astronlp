#!/usr/bin/env python3
import os, json, re, hashlib, random, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm

random.seed(42)

# ---------- 1) Family lexicon (EDIT ME as you learn your corpus) ----------
FAMILY_PATTERNS = {
    "bayesian": [
        r"\bmcmc\b", r"\bbayesian\b", r"\bposterior\b", r"\bprior(s)?\b",
        r"\bmetropolis(-| )?hastings\b", r"\bhamiltonian monte carlo\b",
        r"\bgaussian process(es)?\b", r"\bvariational inference\b", r"\bmarginal likelihood\b"
    ],
    "simulation": [
        r"\bn-?body\b", r"\bsph\b", r"\bhydrodynamic(s|al)?\b",
        r"\bradiative transfer\b", r"\bamr\b", r"\bgadget-?2?\b", r"\benza\b",
        r"\bsimulation setup\b", r"\bhalo catalog(ue)?\b"
    ],
    "lensing": [
        r"\bweak lensing\b", r"\bshear\b", r"\bmass map(ping)?\b", r"\bpsf\b",
        r"\bshape measurement\b", r"\bsource extraction\b"
    ],
    "specphot": [
        r"\bspectral fitting\b", r"\bsed\b", r"\bphotometric redshift(s)?\b", r"\bphoto-?z\b",
        r"\btemplate fitting\b", r"\bcalibration curve\b"
    ],
    "ml_dl": [
        r"\bneural network(s)?\b", r"\bdeep learning\b", r"\bcnn(s)?\b",
        r"\btransformer(s)?\b", r"\brandom forest(s)?\b", r"\bgradient boosting\b", r"\bxgboost\b"
    ],
    # add e.g. "time_series", "spectroscopy", "catalog_matching", etc. as needed
}
FAMILY_REGEX = {fam: [re.compile(p, re.I) for p in pats] for fam, pats in FAMILY_PATTERNS.items()}

# ---------- 2) Cleaning ----------
MATH_PAT = re.compile(r"(\$[^$]+\$)|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)")
CIT_PAT  = re.compile(r"\(([^()]*\d{4}[^()]*)\)|\[[0-9,\s;]+\]")  # (Smith 2019) or [12]
FIG_PAT  = re.compile(r"^\s*(Figure|Table)\s+\d+", re.I)
URL_PAT  = re.compile(r"https?://\S+")

def clean_text(t: str) -> str:
    t = re.sub(MATH_PAT, " ", t or "")
    t = re.sub(CIT_PAT, " <CIT> ", t)
    t = re.sub(URL_PAT, " ", t)
    lines = [ln for ln in t.splitlines() if not re.match(FIG_PAT, ln or "")]
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- 3) Loaders ----------
def load_papers_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            rows.append({
                "paper_id": obj.get("paper_id"),
                "abstract": obj.get("abstract") or "",
                "keywords": [k.lower() for k in (obj.get("keywords") or [])],
            })
    df = pd.DataFrame(rows)
    df["abstract"] = df["abstract"].map(clean_text)
    return df

def split_methodology_paragraphs(raw: str) -> List[str]:
    """
    Robust split: blank lines OR '----' OR paragraphs separated by two newlines.
    Falls back to single lines if file is one-paragraph-per-line.
    """
    if not raw.strip():
        return []
    # Try double-newline split
    parts = re.split(r"\n\s*\n", raw.strip())
    if len(parts) == 1:
        # maybe single lines
        parts = [p for p in raw.splitlines() if p.strip()]
    parts = [clean_text(p) for p in parts]
    return [p for p in parts if len(p) >= 200]  # drop very short fragments

def load_method_paragraphs(txt_dir: Path) -> pd.DataFrame:
    rows = []
    for txt_path in tqdm(sorted(txt_dir.glob("*.txt")), desc="Reading methodology .txt"):
        paper_id = txt_path.stem  # "{paper_id}.txt"
        try:
            raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        paras = split_methodology_paragraphs(raw)
        for i, p in enumerate(paras):
            rows.append({
                "paper_id": paper_id,
                "para_id": f"{paper_id}::p{i:04d}",
                "text": p
            })
    return pd.DataFrame(rows)

# ---------- 4) Family tagging ----------
def detect_family(text: str) -> Optional[str]:
    for fam, regs in FAMILY_REGEX.items():
        if any(r.search(text) for r in regs):
            return fam
    return None

def score_families_from_keywords(keywords: List[str]) -> Dict[str, int]:
    scores = {fam: 0 for fam in FAMILY_REGEX}
    for kw in keywords or []:
        for fam, regs in FAMILY_REGEX.items():
            if any(r.search(kw) for r in regs):
                scores[fam] += 1
    return {k: v for k, v in scores.items() if v > 0}

def families_from_abstract(abstract: str) -> Dict[str, int]:
    scores = {fam: 0 for fam in FAMILY_REGEX}
    for fam, regs in FAMILY_REGEX.items():
        for r in regs:
            hits = len(r.findall(abstract))
            scores[fam] += hits
    return {k: v for k, v in scores.items() if v > 0}

# ---------- 5) Build positives ----------
def build_positive_pairs(
    papers_df: pd.DataFrame,
    paras_df: pd.DataFrame,
    min_conf_per_family: int = 1,
    max_pairs_per_paper_per_family: int = 4,
    max_partners_per_paper_per_family: int = 5,
    require_para_family_match: bool = True,
) -> pd.DataFrame:
    """
    Cross-paper positives within each family.
    Confidence for a paper in a family = keyword_score + abstract_score.
    Only pair papers with confidence >= min_conf_per_family.
    """
    # Compute paper-level family scores
    fam_scores: Dict[str, Dict[str, int]] = {}
    for _, row in papers_df.iterrows():
        kw_scores = score_families_from_keywords(row["keywords"])
        ab_scores = families_from_abstract(row["abstract"])
        merged = {}
        for fam in set(list(kw_scores.keys()) + list(ab_scores.keys())):
            merged[fam] = kw_scores.get(fam, 0) + ab_scores.get(fam, 0)
        if merged:
            fam_scores[row["paper_id"]] = merged

    # Attach paragraph family (optional)
    if require_para_family_match:
        paras_df = paras_df.copy()
        paras_df["family"] = paras_df["text"].map(detect_family)

    pairs = []
    # Index paragraphs by paper (and by family if required)
    by_paper = {pid: g for pid, g in paras_df.groupby("paper_id")}

    # For each family, gather candidate papers with confidence
    for fam in FAMILY_REGEX.keys():
        # papers that “have” this family
        candidate_papers = []
        for pid, scores in fam_scores.items():
            if scores.get(fam, 0) >= min_conf_per_family and pid in by_paper:
                candidate_papers.append((pid, scores.get(fam, 0)))

        if len(candidate_papers) < 2:
            continue

        # Sort by confidence (higher first) to prioritise better anchors
        candidate_papers.sort(key=lambda x: x[1], reverse=True)
        paper_ids = [pid for pid, _ in candidate_papers]

        # Pre-filter paras for this family
        def paras_for(pid):
            df = by_paper[pid]
            if require_para_family_match:
                cand = df[df["family"] == fam]
                if not len(cand):
                    return []
                return list(cand["text"])
            else:
                return list(df["text"])

        # Build cross-paper pairs
        for idx, pid in enumerate(paper_ids):
            A = paras_for(pid)
            if not A:
                continue
            partners = paper_ids[idx+1: idx+1+max_partners_per_paper_per_family]
            random.shuffle(partners)
            made = 0
            for qid in partners:
                B = paras_for(qid)
                if not B:
                    continue
                # sample a handful of combinations
                a_samples = random.sample(A, min(len(A), 2))
                b_samples = random.sample(B, min(len(B), 2))
                for ta in a_samples:
                    tb = random.choice(b_samples)
                    conf = min(fam_scores[pid][fam], fam_scores[qid][fam])
                    pairs.append({
                        "text_a": ta,
                        "text_b": tb,
                        "family": fam,
                        "pos_source": "abstract+keyword",
                        "confidence": int(conf),
                        "paper_id_a": pid,
                        "paper_id_b": qid,
                        # (para ids optional; not tracked here unless stored earlier)
                    })
                    made += 1
                    if made >= max_pairs_per_paper_per_family:
                        break

    if not pairs:
        return pd.DataFrame(columns=[
            "text_a","text_b","family","pos_source","confidence","paper_id_a","paper_id_b"
        ])

    df = pd.DataFrame(pairs)
    # De-dup unordered pairs by text hash
    def key(a,b):
        return hashlib.sha1(("||".join(sorted([a,b]))).encode()).hexdigest()
    df["k"] = [key(a,b) for a,b in zip(df["text_a"], df["text_b"])]
    df = df.drop_duplicates("k").drop(columns=["k"])
    return df

# ---------- 6) Splitting (by paper to avoid leakage) ----------
def split_by_paper(papers_df: pd.DataFrame, train=0.8, val=0.1, test=0.1):
    assert abs(train+val+test - 1.0) < 1e-6
    pids = list(papers_df["paper_id"].unique())
    random.shuffle(pids)
    n = len(pids)
    n_tr = int(n*train); n_va = int(n*val)
    train_ids = set(pids[:n_tr])
    val_ids   = set(pids[n_tr:n_tr+n_va])
    test_ids  = set(pids[n_tr+n_va:])
    return train_ids, val_ids, test_ids

def split_pairs(df_pairs: pd.DataFrame, train_ids, val_ids, test_ids):
    def which_split(pid_a, pid_b):
        if pid_a in train_ids and pid_b in train_ids: return "train"
        if pid_a in val_ids   and pid_b in val_ids:   return "val"
        if pid_a in test_ids  and pid_b in test_ids:  return "test"
        return None  # cross-split pairs get dropped
    split = df_pairs.apply(lambda r: which_split(r.paper_id_a, r.paper_id_b), axis=1)
    out = df_pairs.copy()
    out["split"] = split
    out = out.dropna(subset=["split"])
    return out

# ---------- 7) Main ----------
def main(papers_jsonl: str, paras_dir: str, out_dir: str,
         min_conf: int, max_pairs: int, partners: int, require_para_fam: bool):
    papers_df = load_papers_jsonl(Path(papers_jsonl))
    paras_df  = load_method_paragraphs(Path(paras_dir))
    if papers_df.empty or paras_df.empty:
        raise SystemExit("No papers or paragraphs found. Check paths and filenames.")

    # Keep only papers that have methodology paragraphs
    have_paras = set(paras_df["paper_id"].unique())
    papers_df = papers_df[papers_df["paper_id"].isin(have_paras)].reset_index(drop=True)

    pairs = build_positive_pairs(
        papers_df, paras_df,
        min_conf_per_family=min_conf,
        max_pairs_per_paper_per_family=max_pairs,
        max_partners_per_paper_per_family=partners,
        require_para_family_match=require_para_fam
    )
    if pairs.empty:
        raise SystemExit("No positive pairs built. Relax thresholds or expand FAMILY_PATTERNS.")

    tr_ids, va_ids, te_ids = split_by_paper(papers_df)
    pairs = split_pairs(pairs, tr_ids, va_ids, te_ids)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for split_name in ["train","val","test"]:
        df = pairs[pairs["split"]==split_name].drop(columns=["split"]).reset_index(drop=True)
        outfile = out / f"pairs_{split_name}.jsonl"
        with outfile.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(f"Wrote {len(df):,} pairs -> {outfile}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers_jsonl", required=True, help="Path to papers.jsonl")
    ap.add_argument("--paras_dir", required=True, help="Directory with {paper_id}.txt methodology files")
    ap.add_argument("--out_dir", required=True, help="Output directory for pairs_*.jsonl")
    ap.add_argument("--min_conf", type=int, default=1, help="Min paper family confidence (abstract+keywords)")
    ap.add_argument("--max_pairs", type=int, default=4, help="Max pairs per paper per family")
    ap.add_argument("--partners", type=int, default=5, help="Max partner papers per paper per family")
    ap.add_argument("--require_para_family", action="store_true",
                    help="Require paragraph-level family match (stricter, higher precision)")
    args = ap.parse_args()
    main(args.papers_jsonl, args.paras_dir, args.out_dir,
         args.min_conf, args.max_pairs, args.partners, args.require_para_family)
