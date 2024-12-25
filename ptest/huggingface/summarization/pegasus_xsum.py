from typing import Tuple, List

import torch
from torch import nn
from transformers import pipeline


class ModelWrapper(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        return self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

    def forward(
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past_key_values: List[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        past_kv_list = []
        for i in range(16):
            layers = []
            for j in range(4):
                layers.append(past_key_values[i * 4 + j])
            past_kv_list.append(layers)

        return self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_outputs, ),
            attention_mask=attention_mask,
            past_key_values=tuple(past_kv_list),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

    def forward_init(
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        return self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_outputs, ),
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )


def generate_dummy_past_key_values(num_heads=16,
                                   num_layers=16,
                                   kv_dims=64,
                                   batch_size=1):
    past_key_values = []
    for _ in range(num_layers):
        past_key_values.append(torch.zeros(batch_size, num_heads, 1, kv_dims))
        past_key_values.append(torch.zeros(batch_size, num_heads, 1, kv_dims))
        past_key_values.append(torch.zeros(batch_size, num_heads, 12, kv_dims))
        past_key_values.append(torch.zeros(batch_size, num_heads, 12, kv_dims))

    return past_key_values


def main():
    model_id = "google/pegasus-xsum"
    pipe = pipeline(model=model_id, framework="pt")
    # pipe.model.config.num_beams = 1  # use greedy search
    model = ModelWrapper(pipe.model)

    intput_text = "This is an example input text for summarization."
    output = pipe(intput_text, num_beams=8)
    print(output)

    encoding = pipe.tokenizer(intput_text,
                              return_tensors="pt",
                              max_length=1024,
                              truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    past_key_values = generate_dummy_past_key_values()
    encoder_outputs = model.encode(input_ids, attention_mask)
    # print(decoder_outputs)

    traced_decoder = torch.jit.trace_module(
        model, {
            "encode": [input_ids, attention_mask],
            "forward_init":
            [attention_mask,
             torch.tensor([[0]]), encoder_outputs[0]],
            "forward": [
                attention_mask,
                torch.tensor([[0, 0]]), encoder_outputs[0], past_key_values
            ],
        })
    torch.jit.save(traced_decoder, "model.pt")


if __name__ == '__main__':
    main()
