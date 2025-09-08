#!/usr/bin/env python3 
import json
import logging
import pickle
import torch
import os
from torch.utils.data import Dataset
from typing import List, Dict
from torch import Tensor as T

logger = logging.getLogger(__name__)

def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            logger.info(f"Reading file {path}")
            data = json.load(f)
            results.extend(data)
            logger.info(f"Aggregated data size: {len(results)}")
    return results


# === 1. 恢复 DPR 原函数 ===
def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for path in paths:
        with open(path, "rb") as reader:
            logger.info(f"Reading file {path}")
            data = pickle.load(reader)
            results.extend(data)
            logger.info(f"Aggregated data size: {len(results)}")
    return results

# ✅ 添加 Tensorizer 抽象类（供 hf_models.py 等模块调用）
class Tensorizer:
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True, apply_max_len: bool = True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError


class LogicRetrievalDataset(Dataset):
    """
    用于逻辑检索的自定义数据集
    数据格式：
    {
        "original_question": "...",
        "positive_questions": ["...", "..."],
        "negative_questions": ["...", "..."]
    }
    """

    def __init__(self, file_path: str, tokenizer, seq_length: int = 512):
        super(LogicRetrievalDataset, self).__init__()
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.data = self.load_logic_data(file_path)
        logger.info(f"Loaded {len(self.data)} examples from {file_path}")

    @staticmethod
    def load_logic_data(file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        query = sample["original_question"]
        positive = sample["positive_questions"][0]  # 只取第一个正样本
        negative = sample["negative_questions"][0]  # 只取第一个负样本

        query_encoded = self.tokenizer(
            query, padding='max_length', truncation=True,
            max_length=self.seq_length, return_tensors='pt'
        )
        positive_encoded = self.tokenizer(
            positive, padding='max_length', truncation=True,
            max_length=self.seq_length, return_tensors='pt'
        )
        negative_encoded = self.tokenizer(
            negative, padding='max_length', truncation=True,
            max_length=self.seq_length, return_tensors='pt'
        )

        return {
            "query": {k: v.squeeze(0) for k, v in query_encoded.items()},
            "positive": {k: v.squeeze(0) for k, v in positive_encoded.items()},
            "negative": {k: v.squeeze(0) for k, v in negative_encoded.items()},
        }

def logic_collate_fn(batch):
    queries = {key: torch.stack([item["query"][key] for item in batch]) for key in batch[0]["query"]}
    positives = {key: torch.stack([item["positive"][key] for item in batch]) for key in batch[0]["positive"]}
    negatives = {key: torch.stack([item["negative"][key] for item in batch]) for key in batch[0]["negative"]}
    return queries, positives, negatives
