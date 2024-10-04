#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline


def main():
    pairs = [['what is panda?', 'hi']]

    model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # model = CrossEncoder(model_id, max_length=512)
    # scores = model.predict(pairs)
    # print(scores)
    #
    # model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(pairs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    #
    # model.eval()
    # with torch.no_grad():
    #     scores = model(**inputs, return_dict=True).logits
    #     print(scores)

    model = pipeline(task='text-classification',
                     model=model_id,
                     framework="pt")
    scores = model.predict({"text": "what is panda?", "text_pair": "hi"})

    # inputs = tokenizer.encode_plus("what is panda?", text_pair="hi", return_tensors='pt')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids")

    traced_model = torch.jit.trace(model.model,
                                   (input_ids, attention_mask, token_type_ids),
                                   strict=False)
    torch.jit.save(traced_model, "traced.pt")
    print(scores)


if __name__ == '__main__':
    main()
