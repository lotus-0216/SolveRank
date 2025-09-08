import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch
import torch.nn.functional as F
import faiss
from transformers import BertTokenizer
from dpr.models.biencoder import BiEncoder
from dpr.models.hf_models import HFBertEncoder

def encode_texts(model, tokenizer, texts, device, batch_size=64):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            _, pooled, _ = model(
                input_ids=inputs['input_ids'],
                token_type_ids=inputs['token_type_ids'],
                attention_mask=inputs['attention_mask']
            )
            pooled = F.normalize(pooled, dim=-1)
            all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)

def build_faiss_index(embeddings, use_gpu=True):
    dim = embeddings.shape[1]
    index_cpu = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings.numpy())
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu
    index.add(embeddings.numpy())
    return index

def main():
    pretrained_path = "/home/xxx/solverank/bge-large-en-v1.5"
    model_path = "/home/xxx/solverank/dpr/output/infonce_bge_large_v1.5/infonce_epoch10.pt"
    original_path = "/home/xxx/solverank/dpr/dataset/original_questions.json"
    test_path = "/home/xxx/solverank/dpr/dataset/test_generated_questions_deepseek.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HFBertEncoder.init_tokenizer(pretrained_path)
    encoder = HFBertEncoder.init_encoder(pretrained_path).to(device)
    model = BiEncoder(encoder, encoder).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(test_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    with open(original_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    original_qs = set(item["original_question"].strip() for item in original_data)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    splits = [[all_data[i] for i in test_index] for _, test_index in kf.split(all_data)]

    total_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total_recall = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    total_mrr = 0.0
    total_count = 0
    all_missed = []

    for split_id, split_data in enumerate(splits, 1):
        print(f"\n\U0001F4E6 Evaluating split {split_id} ...")

        retrieval_set = set(original_qs)
        for item in split_data:
            for pos in item["positive_questions"]:
                retrieval_set.add(pos.strip())
        passage_texts = list(retrieval_set)
        passage_embs = encode_texts(model.question_model, tokenizer, passage_texts, device)
        faiss_index = build_faiss_index(passage_embs, use_gpu=True)
        passage_lookup = passage_texts

        hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
        recall_at = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
        reciprocal_ranks = []
        missed_samples = []

        for item in tqdm(split_data, desc=f"Split {split_id}"):
            query = item["original_question"].strip()
            positives = [p.strip() for p in item["positive_questions"]]

            query_emb = encode_texts(model.question_model, tokenizer, [query], device)
            query_np = query_emb.cpu().numpy()
            faiss.normalize_L2(query_np)
            D, I = faiss_index.search(query_np, 200)

            retrieved = []
            for idx in I[0]:
                candidate = passage_lookup[idx]
                if candidate != query:
                    retrieved.append(candidate)
                if len(retrieved) >= 100:
                    break

            hits = [1 if ret in positives else 0 for ret in retrieved]
            num_relevant = len(positives)

            for k in [1, 3, 5, 10]:
                hits_k = hits[:k]
                hits_at[k] += int(any(hits_k))
                recall_at[k] += sum(hits_k) / num_relevant

            try:
                first_hit_rank = next(i + 1 for i, h in enumerate(hits) if h == 1)
                reciprocal_ranks.append(1 / first_hit_rank)
            except StopIteration:
                reciprocal_ranks.append(0)
                missed_samples.append({
                    "query": query,
                    "positive_questions": positives,
                    "retrieved_top10": retrieved[:10]
                })

        count = len(split_data)
        print(f"\n✅ Split {split_id} Results:")
        for k in [1, 3, 5, 10]:
            print(f"Hits@{k}: {hits_at[k] / count:.4f}, Recall@{k}: {recall_at[k] / count:.4f}")
        print(f"MRR:     {sum(reciprocal_ranks) / count:.4f}")

        for k in [1, 3, 5, 10]:
            total_hits[k] += hits_at[k]
            total_recall[k] += recall_at[k]
        total_mrr += sum(reciprocal_ranks)
        total_count += count
        all_missed.extend(missed_samples)

    print("\n\U0001F4CA Overall Evaluation (All Splits):")
    for k in [1, 3, 5, 10]:
        print(f"Hits@{k}: {total_hits[k] / total_count:.4f}, Recall@{k}: {total_recall[k] / total_count:.4f}")
    print(f"MRR:     {total_mrr / total_count:.4f}")

    with open("missed_queries_all.json", "w", encoding="utf-8") as f:
        json.dump(all_missed, f, ensure_ascii=False, indent=2)
    print("❗ Missed queries saved to missed_queries_all.json")

if __name__ == "__main__":
    main()



