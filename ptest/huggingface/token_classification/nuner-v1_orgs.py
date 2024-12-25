#!/usr/bin/env python

from huggingface_hub import snapshot_download
from transformers import pipeline


def main():
    model_id = "guishe/nuner-v1_orgs"
    pipe = pipeline(model=model_id,
                    framework="pt",
                    aggregation_strategy="simple")
    input_text = "Foreign governments may be spying on your smartphone notifications, senator says. Washington (CNN) — Foreign governments have reportedly attempted to spy on iPhone and Android users through the mobile app notifications they receive on their smartphones - and the US government has forced Apple and Google to keep quiet about it, according to a top US senator. Through legal demands sent to the tech giants, governments have allegedly tried to force Apple and Google to turn over sensitive information that could include the contents of a notification - such as previews of a text message displayed on a lock screen, or an update about app activity, Oregon Democratic Sen. Ron Wyden said in a new report. Wyden's report reflects the latest example of long-running tensions between tech companies and governments over law enforcement demands, which have stretched on for more than a decade. Governments around the world have particularly battled with tech companies over encryption, which provides critical protections to users and businesses while in some cases preventing law enforcement from pursuing investigations into messages sent over the internet."

    output = pipe(input_text)
    print(output)

    snapshot_download(repo_id=model_id, local_dir="nuner-v1_orgs")


if __name__ == '__main__':
    main()
