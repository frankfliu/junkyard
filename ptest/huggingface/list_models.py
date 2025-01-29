#!/usr/bin/env python

from collections import defaultdict

from huggingface_hub import HfApi


def main():
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
    main()
