import torch
import torch.nn.functional as F
from torch import nn, Tensor as T
from typing import Tuple, List
from dpr.utils.model_utils import CheckpointState

class BiEncoder(nn.Module):
    """
    修改后的 BiEncoder 模型:
    - 用于问题对问题逻辑检索
    - 使用 InfoNCE 损失函数
    """
    def __init__(self, question_model: nn.Module, fix_q_encoder: bool = False):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.fix_q_encoder = fix_q_encoder

    def get_representation(
        self,
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> T:
        if fix_encoder:
            with torch.no_grad():
                _, pooled_output, _ = sub_model(
                    ids, segments, attn_mask,
                    representation_token_pos=representation_token_pos,
                )
            if sub_model.training:
                pooled_output.requires_grad_(True)
        else:
            _, pooled_output, _ = sub_model(
                ids, segments, attn_mask,
                representation_token_pos=representation_token_pos,
            )
        return pooled_output

    def forward(
        self,
        query_input: dict,
        positive_input: dict,
        negative_input: dict,
        temperature: float = 0.07,
        representation_token_pos=0,
    ) -> T:
        # Get embeddings
        query_repr = self.get_representation(
            self.question_model,
            query_input["input_ids"],
            query_input["token_type_ids"],
            query_input["attention_mask"],
            self.fix_q_encoder,
            representation_token_pos,
        )

        positive_repr = self.get_representation(
            self.question_model,
            positive_input["input_ids"],
            positive_input["token_type_ids"],
            positive_input["attention_mask"],
            self.fix_q_encoder,
            representation_token_pos,
        )

        negative_repr = self.get_representation(
            self.question_model,
            negative_input["input_ids"],
            negative_input["token_type_ids"],
            negative_input["attention_mask"],
            self.fix_q_encoder,
            representation_token_pos,
        )

        # InfoNCE Loss
        pos_sim = torch.exp(torch.sum(query_repr * positive_repr, dim=-1) / temperature)
        neg_sim = torch.exp(torch.matmul(query_repr, negative_repr.t()) / temperature).sum(dim=-1)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()

        return loss

    def create_biencoder_input(
        self,
        samples: List[dict],
        tokenizer,
        seq_length: int = 512,
    ) -> Tuple[dict, dict, dict]:
        query_texts, positive_texts, negative_texts = [], [], []

        for sample in samples:
            query_texts.append(sample["original_question"])
            positive_texts.append(sample["positive_questions"][0])  # 取第一个正样本
            negative_texts.append(sample["negative_questions"][0])  # 取第一个负样本

        queries = tokenizer(query_texts, padding=True, truncation=True, max_length=seq_length, return_tensors='pt')
        positives = tokenizer(positive_texts, padding=True, truncation=True, max_length=seq_length, return_tensors='pt')
        negatives = tokenizer(negative_texts, padding=True, truncation=True, max_length=seq_length, return_tensors='pt')

        return queries, positives, negatives

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()
