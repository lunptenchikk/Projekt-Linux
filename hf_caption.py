# caption/hf_caption.py
import argparse

from pathlib import Path

from PIL import Image
from transformers import pipeline


def main():
    
    ap = argparse.ArgumentParser()
    
    # help dla siebie, zeby sie orientowac
    ap.add_argument("images_dir", help="Folder with images, e.g. caption/images")
    
    ap.add_argument("out_path", help="Output file, e.g. caption/out/captions.txt")
    
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    args = ap.parse_args()



    images_dir = Path(args.images_dir)
    
    
    out_path = Path(args.out_path)
    
    
    
    out_path.parent.mkdir(parents=True, exist_ok=True)


    captioner = pipeline("image-to-text", model=args.model)

    # Typy plikow do obslugi
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    
    
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

    lines = []
    
    
    for p in files:
        
        img = Image.open(p).convert("RGB")
        res = captioner(img)[0]
        
        caption = res.get("generated_text", "").strip()
        
        
        lines.append(f"{p.name} -> {caption}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")





    # Wypisanie do konsoli
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
