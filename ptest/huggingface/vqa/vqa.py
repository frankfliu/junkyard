#!/usr/bin/env python

import requests
import torch
from PIL import Image
from torch import nn
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import pipeline


class VisionModelWrapper(nn.Module):

    def __init__(
        self,
        model,
    ) -> None:
        super().__init__()
        self.model = model

    def forward1(
        self,
        input_ids: torch.LongTensor,
        input_image: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        vision_outputs = self.model.vision_model(
            pixel_values=input_image,
            interpolate_pos_encoding=False,
            return_dict=False)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1],
                                          dtype=torch.long).to(
                                              image_embeds.device)

        question_outputs = self.model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )

        return question_outputs[0]

    def forward(self, question_outputs):
        question_embeds = question_outputs
        question_attention_mask = torch.ones(question_embeds.size()[:-1],
                                             dtype=torch.long)
        bos_ids = torch.tensor([self.model.decoder_start_token_id],
                               dtype=torch.long).repeat(
                                   question_embeds.size(0), 1)
        return self.model.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.model.config.text_config.sep_token_id,
            pad_token_id=self.model.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
        )


def blip():
    model_id = "Salesforce/blip-vqa-base"

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id)

    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url,
                                        stream=True).raw).convert('RGB')

    question = "how many dogs are in the picture?"
    inputs = processor(raw_image, question, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))


def trace_blip():
    model_id = "Salesforce/blip-vqa-base"
    pipe = pipeline("visual-question-answering", model=model_id)
    processor = pipe.image_processor

    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw)

    question = "how many dogs are in the picture?"
    encoding = pipe.tokenizer(question, return_tensors="pt")
    pixel_values = processor(raw_image, return_tensors="pt")
    encoding.update(pixel_values)
    out = pipe.model.generate(**encoding)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    image = encoding["pixel_values"]
    data = (input_ids, image, attention_mask)

    wrapper = VisionModelWrapper(pipe.model)

    question_output = wrapper.forward1(*data)
    out2 = wrapper.forward(question_output)

    visual_module = torch.jit.trace(wrapper, question_output, strict=False)
    torch.jit.save(visual_module, f"blip.pt")

    outputs = {
        "answer": pipe.tokenizer.decode(out2[0],
                                        skip_special_tokens=True).strip()
    }
    print(outputs)


def main():
    model_id = "Salesforce/blip-vqa-base"
    pipe = pipeline("visual-question-answering", model=model_id)

    question = "how many dogs are in the picture?"
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    result = pipe(image, question, top_k=1)
    print(result)

    # inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

    # out = pipe(image)
    #
    # # image = Image.open(image)
    # # image = PIL.ImageOps.exif_transpose(image)
    # # image = image.convert("RGB")
    #
    # # print(out)
    #
    # inputs = torch.zeros((1, 3, 224, 224))
    # pipe.model.config.return_dict = False
    # array = pipe.model(inputs)
    # traced_model = torch.jit.trace(pipe.model, inputs)
    # array2 = traced_model(inputs)
    # torch.jit.save(traced_model, "vit.pt")
    # print(array2[0].shape)


if __name__ == '__main__':
    trace_blip()
    # main()
