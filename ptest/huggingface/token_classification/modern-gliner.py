#!/usr/bin/env python
import os

import torch
from gliner import GLiNER
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
    model_id = "knowledgator/modern-gliner-bi-large-v1.0"
    model = GLiNER.from_pretrained(model_id)

    text = """
    Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
    """

    labels = ["person", "award", "date", "competitions", "teams"]

    entities = model.predict_entities(text, labels, threshold=0.3)

    for entity in entities:
        print(entity["text"], "=>", entity["label"])


def trace():
    model_id = "knowledgator/modern-gliner-bi-large-v1.0"
    pipe = pipeline(model=model_id,
                    framework="pt",
                    aggregation_strategy="simple",
                    device="cpu")

    model = ModelWrapper(pipe.model)
    tokenizer = pipe.tokenizer

    input_text = "Foreign governments may be spying on your smartphone notifications, senator says. Washington (CNN) — Foreign governments have reportedly attempted to spy on iPhone and Android users through the mobile app notifications they receive on their smartphones - and the US government has forced Apple and Google to keep quiet about it, according to a top US senator. Through legal demands sent to the tech giants, governments have allegedly tried to force Apple and Google to turn over sensitive information that could include the contents of a notification - such as previews of a text message displayed on a lock screen, or an update about app activity, Oregon Democratic Sen. Ron Wyden said in a new report. Wyden's report reflects the latest example of long-running tensions between tech companies and governments over law enforcement demands, which have stretched on for more than a decade. Governments around the world have particularly battled with tech companies over encryption, which provides critical protections to users and businesses while in some cases preventing law enforcement from pursuing investigations into messages sent over the internet."

    encoding = tokenizer(input_text, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    traced_model = torch.jit.trace(model, (input_ids, attention_mask))
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.jit.save(traced_model, "models/model.pt")


if __name__ == '__main__':
    main()
    # trace()
