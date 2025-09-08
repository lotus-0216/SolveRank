import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch  
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from dpr.models.biencoder import BiEncoder
from dpr.models.hf_models import HFBertEncoder
import torch.nn.functional as F
import torch.nn as nn
import json
import csv
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import random

logging.basicConfig(filename='infonce_train.log', level=logging.INFO)
logger = logging.getLogger(__name__)


class LogicSupConDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["original_question"]
        positives = item["positive_questions"]
        uid = item["src_uid"]
        return query, positives, uid


def supcon_collate_fn(batch):
    queries, pos_lists, uids = zip(*batch)
    return list(queries), list(pos_lists), list(uids)


class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, contrast_set: torch.Tensor, labels: torch.Tensor):
        anchor = F.normalize(anchor, dim=-1)
        contrast_set = F.normalize(contrast_set, dim=-1)

        sim_scores = torch.matmul(contrast_set, anchor) / self.temperature
        pos_mask = labels == 1
        pos_scores = sim_scores[pos_mask]

        numerator = torch.exp(pos_scores).sum()
        denominator = torch.exp(sim_scores).sum()

        loss = -torch.log(numerator / (denominator + 1e-12))
        return loss


def encode_texts(model, tokenizer, texts, device):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    _, pooled, _ = model(
        input_ids=inputs['input_ids'],
        token_type_ids=inputs['token_type_ids'],
        attention_mask=inputs['attention_mask']
    )
    return pooled


def train_infonce():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    

    pretrained_path = "/home/xxx/solverank/bert-base-uncased"
    data_path = "/home/xxx/solverank/train_generated_questions_deepseek.json"
    rand_neg_path = "/home/xxx/solverank/dpr/dataset/original_questions.json"
    bm25_neg_path = "/home/xxx/solverank/dpr/dataset/negative_questions_bm25.json"
    output_dir = "/home/xxx/solverank/dpr/output/infonce_dpr_rand20_bm5_new0508"
    os.makedirs(output_dir, exist_ok=True)
    loss_log_path = os.path.join(output_dir, "loss_curve.csv")

    # 1. 加载随机负样本全集
    with open(rand_neg_path, 'r', encoding='utf-8') as f:
        neg_question_pool = [item["original_question"].strip() for item in json.load(f)]

    # 2. 加载 bm25 对应的负样本映射
    with open(bm25_neg_path, 'r', encoding='utf-8') as f:
        bm25_neg_map = {
            item["src_uid"]: item["negative_questions"][:5]  # 只取前5个
            for item in json.load(f)
        }

    tokenizer = HFBertEncoder.init_tokenizer(pretrained_path, do_lower_case=True)
    encoder = HFBertEncoder.init_encoder(pretrained_path).to(device)
    biencoder = BiEncoder(encoder, encoder).to(device)

    dataset = LogicSupConDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=supcon_collate_fn, num_workers=0)

    loss_fn = MultiPositiveInfoNCELoss(temperature=0.07)
    optimizer = torch.optim.AdamW(biencoder.parameters(), lr=5e-6)

    loss_values = []

    with open(loss_log_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "step", "loss"])

        biencoder.train()
        for epoch in range(10):
            total_loss = 0
            step_count = 0
            progress = tqdm(loader, desc=f"Epoch {epoch+1}")

            for queries, pos_lists, uids in progress:
                optimizer.zero_grad()

                for i in range(len(queries)):
                    query = queries[i]
                    positives = pos_lists[i][:5]
                    uid = uids[i]

                    # ✅ 构造负样本：20个随机 + 5个BM25
                    rand_negatives = random.sample([q for q in neg_question_pool if q != query], k=20)
                    bm25_negatives = bm25_neg_map.get(uid, [])[:5]
                    full_negatives = rand_negatives + bm25_negatives

                    logger.info("\\n🟡 [DEBUG] Query: %s", query)
                    logger.info("🟢 [DEBUG] Positives: %s", positives)
                    logger.info("🔴 [DEBUG] Negatives: %s", full_negatives)

                    anchor_emb = encode_texts(biencoder.question_model, tokenizer, [query], device)
                    pos_embs = encode_texts(biencoder.question_model, tokenizer, positives, device)
                    neg_embs = encode_texts(biencoder.question_model, tokenizer, full_negatives, device)

                    contrast_set = torch.cat([pos_embs, neg_embs], dim=0)
                    labels = torch.tensor([1] * len(pos_embs) + [0] * len(neg_embs), device=device)

                    loss = loss_fn(anchor_emb.squeeze(0), contrast_set, labels)
                    loss.backward()
                    total_loss += loss.item()
                    step_count += 1

                    writer.writerow([epoch + 1, step_count, loss.item()])
                    loss_values.append(loss.item())

                optimizer.step()
                progress.set_postfix(loss=total_loss / step_count)

            torch.save(biencoder.state_dict(), os.path.join(output_dir, f"infonce_epoch{epoch+1}.pt"))

    # 绘制 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Multi-Positive InfoNCE Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    print("📈 Loss curve saved to:", os.path.join(output_dir, "loss_curve.png"))


if __name__ == '__main__':
    train_infonce()


