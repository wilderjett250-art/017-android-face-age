import argparse
from pathlib import Path

import torch

from train_multitask_fairface import build_model_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("image_size", 128))
    model = build_model_from_config(config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        input_names=["image"],
        output_names=["age_logits", "gender_logits"],
        dynamic_axes={"image": {0: "batch"}, "age_logits": {0: "batch"}, "gender_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported: {output_path}")


if __name__ == "__main__":
    main()
