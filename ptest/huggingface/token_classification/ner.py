import os

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
        return self.model(input_ids, attention_mask)["logits"]


def main():
    model_id = "Jean-Baptiste/roberta-large-ner-english"
    pipe = pipeline(model=model_id, framework="pt", aggregation_strategy="simple", stride=5)
    input_text = "Apple was founded in 1976 by Steve Jobs, Steve Wozniak and Ronald Wayne to develop and sell Wozniak's Apple I personal computer"
    pipe.tokenizer.model_max_length = 10

    output = pipe(input_text)
    print(output)


def trace():
    model_id = "Jean-Baptiste/roberta-large-ner-english"
    pipe = pipeline(model=model_id, framework="pt", aggregation_strategy="simple", device="cpu")

    model = ModelWrapper(pipe.model)
    tokenizer = pipe.tokenizer

    input_text = "Foreign governments may be spying on your smartphone notifications, senator says. Washington (CNN) — Foreign governments have reportedly attempted to spy on iPhone and Android users through the mobile app notifications they receive on their smartphones - and the US government has forced Apple and Google to keep quiet about it, according to a top US senator. Through legal demands sent to the tech giants, governments have allegedly tried to force Apple and Google to turn over sensitive information that could include the contents of a notification - such as previews of a text message displayed on a lock screen, or an update about app activity, Oregon Democratic Sen. Ron Wyden said in a new report. Wyden's report reflects the latest example of long-running tensions between tech companies and governments over law enforcement demands, which have stretched on for more than a decade. Governments around the world have particularly battled with tech companies over encryption, which provides critical protections to users and businesses while in some cases preventing law enforcement from pursuing investigations into messages sent over the internet."

    encoding = tokenizer(input_text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    traced_model = torch.jit.trace(model, (input_ids, attention_mask))
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.jit.save(traced_model, "models/model.pt")


if __name__ == "__main__":
    main()
    # trace()
