import torch
from PIL import Image
from transformers import ViTImageProcessor
from transformers import ViTModel


def main():
    image = Image.open("images/dog_bike_car.jpg")

    model_id = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTModel.from_pretrained(model_id, torchscript=True, return_dict=True)
    # model = ViTModel.from_pretrained(model_id, return_dict=True)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    outputs = model(**inputs)
    print(outputs[0].flatten())

    traced_model = torch.jit.trace(model, [pixel_values])
    torch.jit.save(traced_model, "vit.pt")


if __name__ == "__main__":
    main()
