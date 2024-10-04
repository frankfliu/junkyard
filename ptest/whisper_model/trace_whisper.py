import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def trace():
    model_id = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torchscript=True, attn_implementation="eager")
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # "https://www.voiptroubleshooter.com/open_speech/chinese/OSR_cn_000_0072_8k.wav"
    audio_path = "OSR_cn_000_0072_8k.wav"
    audio, sample_rate = librosa.load(audio_path,
                                      sr=16000)  # Resample audio to 16000 Hz

    input_features = processor(audio, return_tensors="pt").input_features

    # input_features = processor(np.concatenate(test), return_tensors="pt").input_features
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids,
                                           skip_special_tokens=True)[0]
    print(f"===== Original: {transcription}")

    # Start tracing
    traced_model = torch.jit.trace_module(model,
                                          {"generate": [input_features]})
    torch.jit.save(traced_model, "traced_whisper.pt")
    generated_ids = traced_model.generate(input_features)
    print(f"===== generated_ids: {generated_ids}")

    transcription = processor.batch_decode(generated_ids,
                                           skip_special_tokens=True)[0]
    print(f"===== Traced: {transcription}")
    # processor.tokenizer.save_pretrained("whisper-tokenizer")


if __name__ == '__main__':
    trace()
