#!/usr/bin/env python

import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import hf_hub_download
from torch import nn


class ModelWrapper(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model.model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        last_hidden_state = self.model.model(
            input_ids, attention_mask, return_dict=True).last_hidden_state
        dense_vecs = self.model._dense_embedding(last_hidden_state,
                                                 attention_mask)
        dense_vecs = F.normalize(dense_vecs, dim=-1)

        sparse_vecs = self.model._sparse_embedding(last_hidden_state,
                                                   input_ids,
                                                   return_embedding=False)
        return {
            "dense_vecs": dense_vecs,
            "sparse_vecs": sparse_vecs,
        }


def main():
    model_id = "BAAI/bge-m3"
    sentences = ["What is BGE M3?", "Definition of BM25"]

    model = BGEM3FlagModel(model_id, devices=["cpu"])
    output_1 = model.encode(sentences,
                            return_dense=True,
                            return_sparse=True,
                            return_colbert_vecs=False)
    print(output_1)

    wrapper = ModelWrapper(model)
    embedding = model.tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = embedding["input_ids"]
    attention_mask = embedding["attention_mask"]
    output_2 = wrapper(input_ids, attention_mask)
    print(output_2)

    sparse_vecs = output_2["sparse_vecs"]

    unused_tokens = set()
    for _token in ['cls_token', 'eos_token', 'pad_token', 'unk_token']:
        if _token in model.tokenizer.special_tokens_map:
            _token_id = model.tokenizer.convert_tokens_to_ids(
                model.tokenizer.special_tokens_map[_token])
            unused_tokens.add(_token_id)

    all_lexical_weights = []
    all_token_weights = sparse_vecs.squeeze(-1).cpu().detach().numpy()
    input_ids = input_ids.numpy()

    for token_weights, ids in zip(all_token_weights, input_ids):
        result = defaultdict(int)
        for w, idx in zip(token_weights, ids):
            if idx not in unused_tokens and w > 0:
                idx = str(idx)
                if w > result[idx]:
                    result[idx] = w

        all_lexical_weights.append(result)

    print(all_lexical_weights)


def save_serving_properties(model_id: str):
    model_name = model_id.split("/")[1]

    arguments = {
        "engine": "PyTorch",
        "option.modelName": model_name,
        "option.mapLocation": "true",
        "padding": "true",
    }
    file = hf_hub_download(repo_id=model_id,
                           filename="sentence_bert_config.json")
    with open(file) as f:
        config = json.load(f)
        if config.get("max_seq_length"):
            arguments["maxLength"] = config.get("max_seq_length")
        if config.get("do_lower_case"):
            arguments["doLowerCase"] = config.get("do_lower_case")

    serving_file = os.path.join(model_name, "serving.properties")
    with open(serving_file, 'w') as f:
        for k, v in arguments.items():
            f.write(f"{k}={v}\n")


def jit_trace():
    model_id = "BAAI/bge-m3"
    sentences = ["What is BGE M3?", "Definition of BM25"]

    model = BGEM3FlagModel(model_id, devices=["cpu"])

    embedding = model.tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = embedding["input_ids"]
    attention_mask = embedding["attention_mask"]

    traced_model = torch.jit.trace(ModelWrapper(model),
                                   (input_ids, attention_mask),
                                   strict=False)

    model_name = model_id.split("/")[1]
    os.makedirs(model_name, exist_ok=True)

    model.tokenizer.save_pretrained(model_name)
    for path in os.listdir(model_name):
        if path != "tokenizer.json" and path != "tokenizer_config.json" and path != "special_tokens_map.json":
            os.remove(os.path.join(model_name, path))

    torch.jit.save(traced_model, f"{model_name}/{model_name}.pt")

    save_serving_properties(model_id)


if __name__ == '__main__':
    main()
    # jit_trace()
