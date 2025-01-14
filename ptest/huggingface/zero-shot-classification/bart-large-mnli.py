import os.path

import torch
from torch import nn
from transformers import pipeline


class ModelWrapper(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask)[0]


def main():
    model_id = "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification", model=model_id)
    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    output = classifier(sequence_to_classify, candidate_labels)
    print(output)

    candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
    output = classifier(sequence_to_classify,
                        candidate_labels,
                        multi_label=True)
    print(output)


def pytorch_prediction():
    model_id = "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification",
                          model=model_id,
                          device="cpu")
    model = classifier.model
    tokenizer = classifier.tokenizer

    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    for label in candidate_labels:
        hypothesis = f'This example is {label}.'

        encoding = tokenizer(sequence_to_classify,
                             hypothesis,
                             return_tensors='pt',
                             truncation_strategy='only_first')
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        output = model(input_ids, attention_mask)

        logits = output[0]

        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        print(prob_label_is_true)


def trace():
    model_id = "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification",
                          model=model_id,
                          device="cpu")

    model = ModelWrapper(classifier.model.model)
    tokenizer = classifier.tokenizer

    sequence_to_classify = "one day I will see the world"
    hypothesis = f'This example is travel.'

    encoding = tokenizer(sequence_to_classify,
                         hypothesis,
                         return_tensors='pt',
                         truncation_strategy='only_first')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    traced_model = torch.jit.trace(model, (input_ids, attention_mask))

    model_dir = model_id.split("/")[1]
    os.makedirs(model_dir, exist_ok=True)
    torch.jit.save(traced_model, f"{model_dir}/model.pt")


if __name__ == '__main__':
    main()
    # pytorch_prediction()
    # trace()
