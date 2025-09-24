
#!/usr/bin/env python3
"""
Build contrastive positive pairs for SPECTER2 adapter training using a tuned lexicon.

Inputs:
  --papers_jsonl  : JSONL with fields {paper_id, abstract, keywords[list]}
  --paras_dir     : directory with methodology text files named {paper_id}.txt
  --lexicon_json  : path to lexicon_v2.json (regex patterns per family)
  --out_dir       : where pairs_{train,val,test}.jsonl will be written

Key flags:
  --use_abstracts           : also scan abstracts with family regex (slower, but better coverage)
  --min_conf <int>          : minimum family "confidence" per paper (sum of matches) to include (default: 1)
  --require_para_family     : only use paragraphs that themselves match the family (higher precision)
  --max_pairs <int>         : max pairs per paper per family (default: 4)
  --partners <int>          : max partner-papers per paper per family (default: 5)
  --split 0.8 0.1 0.1       : train/val/test ratios (sum to 1.0)

Outputs (JSONL lines):
  {"text_a":..., "text_b":..., "family":..., "pos_source":"abstract+keyword",
   "confidence": <int>, "paper_id_a":..., "paper_id_b":..., "para_id_a":..., "para_id_b":...}

Usage example:
  python pair_builder_with_lexicon.py \
    --papers_jsonl /path/to/papers.jsonl \
    --paras_dir /path/to/methodology_txts \
    --lexicon_json /path/to/lexicon_v2.json \
    --out_dir /path/to/pairs_out \
    --use_abstracts \
    --require_para_family \
    --min_conf 1 --max_pairs 4 --partners 5
"""
import json, re, argparse, random, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict

random.seed(42)

# ---------------------- IO ----------------------
def load_jsonl_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def load_papers_jsonl(path: Path):
    rows = []
    for obj in load_jsonl_rows(path):
        rows.append({
            "paper_id": obj.get("paper_id"),
            "abstract": obj.get("abstract") or "",
            "keywords": [normalize_kw(k) for k in (obj.get("keywords") or [])],
        })
    return rows

def load_lexicon(path: Path) -> Dict[str, List[str]]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

# ---------------------- Text utils ----------------------
KW_KEEP = re.compile(r"[^a-z0-9+\-.\s]")
DASHES = re.compile(r"[\u2010-\u2015]")  # various unicode dashes
SPC = re.compile(r"\s+")

def normalize_kw(s: str) -> str:
    s = (s or "").lower().strip()
    s = DASHES.sub("-", s)
    s = KW_KEEP.sub(" ", s)
    s = SPC.sub(" ", s).strip()
    return s

MATH_PAT = re.compile(r"(\$[^$]+\$)|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)")
CIT_PAT  = re.compile(r"\(([^()]*\d{4}[^()]*)\)|\[[0-9,\s;]+\]")
FIG_PAT  = re.compile(r"^\s*(Figure|Table)\s+\d+", re.I)
URL_PAT  = re.compile(r"https?://\S+")

def clean_text(t: str) -> str:
    t = re.sub(MATH_PAT, " ", t or "")
    t = re.sub(CIT_PAT, " <CIT> ", t)
    t = re.sub(URL_PAT, " ", t)
    lines = [ln for ln in (t or '').splitlines() if not re.match(FIG_PAT, ln or "")]
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_paragraphs(raw: str) -> List[str]:
    if not raw or not raw.strip(): return []
    parts = re.split(r"\n\s*\n", raw.strip())
    if len(parts) == 1:
        parts = [p for p in raw.splitlines() if p.strip()]
    parts = [clean_text(p) for p in parts]
    return [p for p in parts if len(p) >= 200]

def load_method_paragraphs(txt_dir: Path):
    rows = []
    for p in sorted(txt_dir.glob("*.txt")):
        pid = p.stem
        try:
            raw = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        paras = split_paragraphs(raw)
        for i, txt in enumerate(paras):
            rows.append({"paper_id": pid, "para_id": f"{pid}::p{i:04d}", "text": txt})
    return rows

# ---------------------- Family detection ----------------------
def compile_family_regex(family_patterns: Dict[str, List[str]]):
    # compile one big OR regex per family for speed
    comp = {}
    for fam, pats in family_patterns.items():
        if not pats: continue
        big = "(?:" + "|".join(pats) + ")"
        comp[fam] = re.compile(big, re.I)
    return comp

def families_in_text(text: str, comp_map: Dict[str, re.Pattern]) -> Set[str]:
    hits = set()
    for fam, rgx in comp_map.items():
        if rgx.search(text):
            hits.add(fam)
    return hits

def family_scores_from_keywords(keywords: List[str], comp_map: Dict[str, re.Pattern]) -> Dict[str, int]:
    scores = Counter()
    for kw in set(keywords):
        for fam, rgx in comp_map.items():
            if rgx.search(kw):
                scores[fam] += 1
    return dict(scores)

def family_scores_from_abstract(abstract: str, comp_map: Dict[str, re.Pattern]) -> Dict[str, int]:
    scores = {}
    for fam, rgx in comp_map.items():
        hits = len(rgx.findall(abstract or ""))
        if hits:
            scores[fam] = hits
    return scores

# ---------------------- Pairs ----------------------
def build_pairs(papers, paras, comp_map,
                use_abstracts: bool = False,
                min_conf: int = 1,
                require_para_family: bool = True,
                max_pairs_per_paper_per_family: int = 4,
                max_partners_per_paper_per_family: int = 5):
    # Index paragraphs by paper
    by_paper = {}
    for r in paras:
        by_paper.setdefault(r["paper_id"], []).append(r)

    # Paper-level family scores
    paper_fams = {}      # pid -> {fam: score}
    for p in papers:
        pid = p["paper_id"]
        if pid not in by_paper:
            continue
        kw_scores = family_scores_from_keywords(p["keywords"], comp_map)
        if use_abstracts:
            ab_scores = family_scores_from_abstract(p["abstract"], comp_map)
        else:
            ab_scores = {}
        fams = Counter(kw_scores) + Counter(ab_scores)
        fams = {f: int(s) for f, s in fams.items() if s >= min_conf}
        if fams:
            paper_fams[pid] = fams

    # Precompute paragraph families (only if needed)
    para_fams_cache = {}
    if require_para_family:
        for r in paras:
            fams = families_in_text(r["text"], comp_map)
            if fams:
                para_fams_cache[r["para_id"]] = fams

    # Build cross-paper positives within each family
    pairs = []
    fam_to_papers = defaultdict(list)  # fam -> list of (pid, score)
    for pid, fams in paper_fams.items():
        for fam, sc in fams.items():
            fam_to_papers[fam].append((pid, sc))
    for fam, plist in fam_to_papers.items():
        if len(plist) < 2:
            continue
        # sort by score, high first
        plist.sort(key=lambda x: x[1], reverse=True)
        paper_ids = [pid for pid, _ in plist]

        def paras_for(pid):
            items = by_paper.get(pid, [])
            if not require_para_family:
                return items
            # filter to paragraphs that include this family
            out = []
            for it in items:
                fams = para_fams_cache.get(it["para_id"])
                if fams and fam in fams:
                    out.append(it)
            return out

        for i, pid in enumerate(paper_ids):
            A = paras_for(pid)
            if not A:
                continue
            partners = paper_ids[i+1 : i+1+max_partners_per_paper_per_family]
            random.shuffle(partners)
            made = 0
            for qid in partners:
                B = paras_for(qid)
                if not B:
                    continue
                # sample a few combos
                a_samp = random.sample(A, min(len(A), 2))
                b_samp = random.sample(B, min(len(B), 2))
                for ra in a_samp:
                    rb = random.choice(b_samp)
                    conf = min(paper_fams[pid][fam], paper_fams[qid][fam])
                    pairs.append({
                        "text_a": ra["text"], "text_b": rb["text"],
                        "family": fam, "pos_source": "abstract+keyword",
                        "confidence": int(conf),
                        "paper_id_a": pid, "paper_id_b": qid,
                        "para_id_a": ra["para_id"], "para_id_b": rb["para_id"],
                    })
                    made += 1
                    if made >= max_pairs_per_paper_per_family:
                        break
    # Dedup unordered text pairs
    def key(a,b):
        return hashlib.sha1(("||".join(sorted([a,b]))).encode()).hexdigest()
    seen = set()
    uniq = []
    for r in pairs:
        k = key(r["text_a"], r["text_b"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq

# ---------------------- Splitting ----------------------
def split_by_paper(papers, train=0.8, val=0.1, test=0.1):
    assert abs(train+val+test - 1.0) < 1e-6
    pids = [p["paper_id"] for p in papers]
    random.shuffle(pids)
    n = len(pids); n_tr = int(n*train); n_va = int(n*val)
    train_ids = set(pids[:n_tr])
    val_ids   = set(pids[n_tr:n_tr+n_va])
    test_ids  = set(pids[n_tr+n_va:])
    return train_ids, val_ids, test_ids

def assign_split(pair, tr, va, te):
    a, b = pair["paper_id_a"], pair["paper_id_b"]
    if a in tr and b in tr: return "train"
    if a in va and b in va: return "val"
    if a in te and b in te: return "test"
    return None

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers_jsonl", required=True)
    ap.add_argument("--paras_dir", required=True)
    ap.add_argument("--lexicon_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_abstracts", action="store_true")
    ap.add_argument("--min_conf", type=int, default=1)
    ap.add_argument("--require_para_family", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=4)
    ap.add_argument("--partners", type=int, default=5)
    ap.add_argument("--split", type=float, nargs=3, default=[0.8,0.1,0.1])
    args = ap.parse_args()

    papers = load_papers_jsonl(Path(args.papers_jsonl))
    paras  = load_method_paragraphs(Path(args.paras_dir))
    if not papers or not paras:
        raise SystemExit("No papers or paragraphs found. Check inputs.")

    # keep only papers with paragraphs
    have = {r["paper_id"] for r in paras}
    papers = [p for p in papers if p["paper_id"] in have]

    lex = load_lexicon(Path(args.lexicon_json))
    comp_map = compile_family_regex(lex)

    pairs = build_pairs(
        papers, paras, comp_map,
        use_abstracts=args.use_abstracts,
        min_conf=args.min_conf,
        require_para_family=args.require_para_family,
        max_pairs_per_paper_per_family=args.max_pairs,
        max_partners_per_paper_per_family=args.partners
    )

    tr, va, te = split_by_paper(papers, *args.split)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    for r in pairs:
        s = assign_split(r, tr, va, te)
        if s: splits[s].append(r)

    for name, items in splits.items():
        p = out / f"pairs_{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(items):,} pairs -> {p}")

if __name__ == "__main__":
    main()
