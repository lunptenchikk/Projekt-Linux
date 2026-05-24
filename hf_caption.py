import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("out_path")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

    lines = []

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)

        caption = processor.decode(output[0], skip_special_tokens=True)
        line = f"{img_path.name} -> {caption}"
        lines.append(line)
        print(line)

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()