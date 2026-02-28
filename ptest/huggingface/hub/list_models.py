#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Optional

from datasets import load_dataset

import requests
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

allowed_actions = ["list_models", "download_model", "list_datasets", "save_dataset"]


def print_pooling_mode():
    category = "sentence-similarity"
    api = HfApi()
    models = api.list_models(filter=f"{category},pytorch", sort="downloads", direction=-1)
    if not models:
        logging.warning(f"no model matches category: {category}.")

    logging.info(f"total models: {len(models)}")
    for model_info in models:
        model_id = model_info.modelId
        try:
            file = hf_hub_download(repo_id=model_id, filename="1_Pooling/config.json")
            if os.path.exists(file):
                with open(file, "r") as f:
                    pooling = json.load(f)
                    if pooling.get("pooling_mode_max_tokens"):
                        logging.info(f"{model_id}: max")
                    elif pooling.get("pooling_mode_mean_sqrt_len_tokens"):
                        logging.info(f"{model_id}: mean_sqrt_len")
                    elif pooling.get("pooling_mode_cls_token"):
                        # logging.info(f"{model_id}: cls")
                        pass
                    elif pooling.get("pooling_mode_mean_tokens"):
                        # logging.info(f"{model_id}: mean")
                        pass
                    elif pooling.get("pooling_mode_weightedmean_tokens"):
                        logging.info(f"{model_id}: weightedmean")
                    elif pooling.get("pooling_mode_lasttoken"):
                        logging.info(f"{model_id}: lasttoken")
        except requests.exceptions.HTTPError:
            pass


def download_model(model_id: str):
    snapshot_download(repo_id=model_id, local_dir=model_id)


def list_models(search: str):
    api = HfApi()
    models = api.list_models(filter=search, sort="downloads", direction=-1, limit=100)

    tasks = defaultdict(list)
    for model_info in models:
        model_id = model_info.modelId
        tag = model_info.pipeline_tag
        tasks[tag].append(model_id)

    for k, v in tasks.items():
        for model_id in v:
            print(f"{k}: {model_id}")


def list_datasets(search: str):
    api = HfApi()
    for info in api.list_datasets(search=search, limit=100):
        print(info.id)


def save_dataset(repo_id: str, name: Optional[str] = None, split: str = "test", revision: Optional[int] = None):
    file_name = f"{split}{revision}" if revision else f"{split}"
    cache_file = os.path.expanduser(f"~/.cache/datasets/{repo_id}/{name}/{file_name}.jsonl")
    if os.path.isfile(cache_file):
        return

    dataset = load_dataset(repo_id, name=name, split=split, revision=revision)
    dataset.to_json(cache_file, lines=True, orient="records")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, choices=allowed_actions, required=True)
    parser.add_argument("--search", type=str, help="search string")
    parser.add_argument("--repo-id", type=str, help="repo id")
    parser.add_argument("--name", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--revision", type=int)
    args = parser.parse_args()
    if args.action == "list_models":
        list_models(args.search if args.search else "modernbert")
    elif args.action == "download_model":
        download_model(args.repo_id)
    elif args.action == "list_datasets":
        list_models(args.search if args.search else "HuggingFaceH4/MATH-500")
    elif args.action == "save_dataset":
        # save_dataset("cais/mmlu", "abstract_algebra")
        save_dataset(args.repo_id, args.name, args.split, args.revision)
    else:
        raise ValueError(f"Unknown action: {args.action}")
