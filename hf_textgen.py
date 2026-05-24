# textgen/hf_textgen.py


import argparse

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(): 
    
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--model", default="distilgpt2")
    
    ap.add_argument("--seed", default="The ")
    
    ap.add_argument("--len", type=int, default=200)
    
    ap.add_argument("--method", choices=["greedy", "temp", "beam"], default="temp")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--beam", type=int, default=3)
    
    
    
    ap.add_argument("--out", default=None, help="Optional sciezka do zapisu wygenerowanego tekstu")
    
    
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"





 # Pobranie modelu i tokenizera
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    

    inputs = tok(args.seed, return_tensors="pt").to(device)


    gen_kwargs = {
        "max_new_tokens": args.len,
        "pad_token_id": tok.eos_token_id,
    }



# Generowanie tekstu wg wybranej metody

    if args.method == "greedy":
        
        gen_kwargs.update({"do_sample": False})
        
    elif args.method == "temp":
        
        
        
        gen_kwargs.update({"do_sample": True, "temperature": args.temp})
        
        
        
    elif args.method == "beam":
        
        gen_kwargs.update({"do_sample": False, "num_beams": args.beam, "early_stopping": True})

    with torch.no_grad():
        
        # Generowanie tekstu
        out_ids = model.generate(**inputs, **gen_kwargs)

    text = tok.decode(out_ids[0], skip_special_tokens=True)

    # Wypisanie wygenerowanego tekstu
    print(text)


    if args.out:
        
        out_path = Path(args.out)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        
        
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
