# #!/usr/bin/env python3
# import json, random, argparse
# from pathlib import Path
# from sentence_transformers import SentenceTransformer, util

# def load_pairs(path: Path, max_n=None):
#     rows = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             o = json.loads(line)
#             rows.append((o["text_a"], o["text_b"]))
#     random.shuffle(rows)
#     if max_n:
#         rows = rows[:max_n]
#     return rows

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pairs_jsonl", required=True)
#     ap.add_argument("--model_path", required=True, help="Trained model dir")
#     ap.add_argument("--base_model", default=None, help="Also eval base model for comparison (e.g., allenai/specter2_base)")
#     args = ap.parse_args()

#     pairs = load_pairs(Path(args.pairs_jsonl))
#     A = [a for a, b in pairs]
#     B = [b for a, b in pairs]
#     random_pairs = list(zip(A, random.sample(B, len(B))))

#     def eval_model(mpath_or_id):
#         model = SentenceTransformer(mpath_or_id)
#         va = model.encode(A, convert_to_tensor=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
#         vb = model.encode(B, convert_to_tensor=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
#         vr_b = model.encode([b for _, b in random_pairs], convert_to_tensor=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
#         cos_pos = util.cos_sim(va, vb).diagonal().cpu().numpy()
#         cos_rand = util.cos_sim(va, vr_b).diagonal().cpu().numpy()
#         return cos_pos, cos_rand

#     pos, rnd = eval_model(args.model_path)
#     print(f"Trained model — mean cosine: positives={pos.mean():.4f}, random={rnd.mean():.4f}")

#     if args.base_model:
#         pos_b, rnd_b = eval_model(args.base_model)
#         print(f"Base model — mean cosine: positives={pos_b.mean():.4f}, random={rnd_b.mean():.4f}")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
import os, json, random, argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, models, util
import adapters  # <-- important

ADAPTER_NAME = "astro_methods"

def load_pairs(path: Path, max_n=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            rows.append((o["text_a"], o["text_b"]))
    random.shuffle(rows)
    if max_n:
        rows = rows[:max_n]
    return rows

def load_st_with_adapter(trained_dir: str, base_id: str = "allenai/specter2_base", max_seq_len: int = 256):
    """
    Rebuild the same ST architecture used in training (Transformer + Pooling),
    then load + activate the saved adapter from trained_dir/adapter/.
    """
    word = models.Transformer(base_id, max_seq_length=max_seq_len)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st = SentenceTransformer(modules=[word, pool])

    adapters.init(word.auto_model)
    adapter_dir = os.path.join(trained_dir, "adapter")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter folder not found: {adapter_dir}")
    word.auto_model.load_adapter(adapter_dir, load_as=ADAPTER_NAME)
    word.auto_model.set_active_adapters(ADAPTER_NAME)
    return st

def eval_model_embedder(model):
    def enc(texts):
        return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--model_path", required=True, help="Path to trained model dir (that contains adapter/)")
    ap.add_argument("--base_model", default="allenai/specter2_base", help="Baseline for comparison")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs_jsonl))
    A = [a for a,b in pairs]
    B = [b for a,b in pairs]
    random_pairs = list(zip(A, random.sample(B, len(B))))

    # Trained adapter model
    trained = load_st_with_adapter(args.model_path, base_id=args.base_model)
    enc_tr = eval_model_embedder(trained)
    va, vb, vrb = enc_tr(A), enc_tr(B), enc_tr([b for _, b in random_pairs])
    pos = util.cos_sim(va, vb).diagonal().cpu().numpy()
    rnd = util.cos_sim(va, vrb).diagonal().cpu().numpy()
    print(f"Trained+adapter — mean cosine: positives={pos.mean():.4f}, random={rnd.mean():.4f}")

    # Pure base (no adapter) for reference
    base = SentenceTransformer(args.base_model)  # ST will wrap with mean pooling
    enc_base = lambda X: base.encode(X, convert_to_tensor=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    va_b, vb_b, vrb_b = enc_base(A), enc_base(B), enc_base([b for _, b in random_pairs])
    pos_b = util.cos_sim(va_b, vb_b).diagonal().cpu().numpy()
    rnd_b = util.cos_sim(va_b, vrb_b).diagonal().cpu().numpy()
    print(f"Base only      — mean cosine: positives={pos_b.mean():.4f}, random={rnd_b.mean():.4f}")

if __name__ == "__main__":
    main()
