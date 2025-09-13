from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-seg.pt")  # load an official model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    print(len(results))
    for result in results:
        _xy = result.masks.xy  # mask in polygon format
        _xyn = result.masks.xyn  # normalized
        _masks = result.masks.data  # mask in matrix format (num_objects x H x W)

        # print(result)


if __name__ == "__main__":
    main()
