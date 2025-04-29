import torch
from TTS.api import TTS


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(TTS().list_models())

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    # tts = TTS("tts_models/en/multi-dataset/tortoise-v2").to(device)

    # wav = tts.tts(text="Hello world!", speaker_wav="LJ001-0001.wav", language="en")
    # Text to speech to a file
    tts.tts_to_file(text="Hello world!",
                    speaker_wav="LJ001-0001.wav",
                    language="en",
                    file_path="output.wav")


if __name__ == '__main__':
    main()
