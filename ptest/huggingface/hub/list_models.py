#!/usr/bin/env python

import json
import logging
import os
import sys

import requests
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download


def print_pooling_mode():
    category = "sentence-similarity"
    api = HfApi()
    models = api.list_models(filter=f"{category},pytorch",
                             sort="downloads",
                             direction=-1)
    if not models:
        logging.warning(f"no model matches category: {category}.")

    logging.info(f"total models: {len(models)}")
    for model_info in models:
        model_id = model_info.modelId
        try:
            file = hf_hub_download(repo_id=model_id,
                                   filename="1_Pooling/config.json")
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


def list_models():
    api = HfApi()
    models = api.list_models(filter="modernbert",
                             sort="downloads",
                             direction=-1,
                             limit=100)

    tasks = defaultdict(list)
    for model_info in models:
        model_id = model_info.modelId
        tag = model_info.pipeline_tag
        tasks[tag].append(model_id)

    for k, v in tasks.items():
        for model_id in v:
            print(f"{k}: {model_id}")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)
    list_models()
