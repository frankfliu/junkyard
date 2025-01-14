from typing import Tuple, List

import torch
from torch import nn
from transformers import AutoTokenizer, MBartForConditionalGeneration


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
        for i in range(12):
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
                                   num_layers=12,
                                   kv_dims=64,
                                   batch_size=1):
    past_key_values = []
    for _ in range(num_layers):
        past_key_values.append(torch.zeros(batch_size, num_heads, 1, kv_dims))
        past_key_values.append(torch.zeros(batch_size, num_heads, 1, kv_dims))
        past_key_values.append(torch.zeros(batch_size, num_heads, 21, kv_dims))
        past_key_values.append(torch.zeros(batch_size, num_heads, 21, kv_dims))

    return past_key_values


def jit_trace():
    model_id = "facebook/mbart-large-50-many-to-many-mmt"
    mbart = MBartForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              src_lang="fr_XX",
                                              tgt_lang="en_XX")
    model = ModelWrapper(mbart)

    input_text = "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militai"

    encoding = tokenizer(input_text, return_tensors="pt")
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

    model_dir = model_id.split("/")[1]
    os.makedirs(model_dir, exist_ok=True)
    torch.jit.save(traced_decoder, f"{model_dir}/model.pt")


def main():
    model_id = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              src_lang="fr_XX",
                                              tgt_lang="en_XX")
    # tokenizer.save_pretrained("output/mbart-large")
    # tokenizer = AutoTokenizer.from_pretrained("mbart-large",
    #                                           src_lang="fr_XX",
    #                                           tgt_lang="en_XX")
    input_text = "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militai"

    model_inputs = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(
        **model_inputs,
        num_beams=2,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    print(output)


def convert2onnx():
    model_id = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              src_lang="fr_XX",
                                              tgt_lang="en_XX")

    input_text = "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militai"
    model_inputs = tokenizer(input_text, return_tensors="pt")

    onnx_path = "output/mbart-large-50-many-to-many-mmt.onnx"

    # Export to ONNX format:
    torch.onnx.export(
        model, (model_inputs["input_ids"], model_inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {
            0: "batch",
            1: "sequence"
        }},
        opset_version=14)


if __name__ == '__main__':
    # jit_trace()
    main()
    # convert2onnx()
