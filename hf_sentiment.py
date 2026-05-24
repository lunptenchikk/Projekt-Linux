# sentiment/hf_sentiment.py

import argparse

from pathlib import Path

from transformers import pipeline


def label_to_pm_one(label: str) -> int:
    
    
    label = label.upper().strip() # Noramlizacja etykiety
    
    if label in ("POSITIVE", "LABEL_1"):
        return 1
    if label in ("NEGATIVE", "LABEL_0"):
        return -1
    return -1


def find_file(test_dir: Path, rec_id: str) -> Path | None:
    rec_id = rec_id.strip()
    rec_id = Path(rec_id).name  # zostawiamy tylko nazwe pliku

    candidates = [ # na wszelki wypadek sprawdzamy kilka mozliwych rozszerzen
        
        test_dir / rec_id,
        test_dir / (rec_id + ".txt"),
        test_dir / (rec_id + ".dat"),
    ]
    
    
    for p in candidates:
        if p.exists() and p.is_file():
            return p
        
        
    return None


def main():
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("test_dir", help="Folder with files like rec_375 (e.g. test_rec)")
    
    
    ap.add_argument("test_list", help="Path to oceny_test_rec.out (lines: rec_375 -1)")
    
    
    ap.add_argument("out_path", help="Output oceny_student.out (only labels per line)")
    
    
    ap.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    ap.add_argument("--batch", type=int, default=16)
    
    
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()


# Przygotowanie sciezek

    test_dir = Path(args.test_dir)
    test_list = Path(args.test_list)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    # Wczytanie listy plikow i etykiet
    pairs = []
    
    for line in test_list.read_text(encoding=args.encoding, errors="ignore").splitlines():
        
        line = line.strip()
        
        if not line:
            
            continue
        
        parts = line.split()
        
        rec_id = parts[0]
        
        y_true = None
        
        if len(parts) >= 2:
            
            try:
                y_true = int(parts[1])
            except ValueError:
                y_true = None
                
                
        pairs.append((rec_id, y_true))

    # Teksty do analizy
    texts = []
    
    exists_mask = []
    
    for rec_id, _ in pairs:
        
        p = find_file(test_dir, rec_id)
        
        if p is None:
            
            texts.append(None)
            exists_mask.append(False)
            
        else:
            
            texts.append(p.read_text(encoding=args.encoding, errors="ignore"))
            exists_mask.append(True)

    # Klasyfikator sentymentu
    
    clf = pipeline("sentiment-analysis", model=args.model)

# Na podstawie exists_mask wybieramy tylko istniejace pliki
    valid_idx = [i for i, ok in enumerate(exists_mask) if ok]
    
    valid_texts = [texts[i] for i in valid_idx]
    
    
    
    # Predykcje dla istniejacych plikow
    preds = clf(valid_texts, batch_size=args.batch, truncation=True) if valid_texts else []





    # Ustawiamy -1 jako default dla brakujacych plikow
    y_pred = [-1] * len(pairs)
    
    for i, pred in zip(valid_idx, preds):
        
        y_pred[i] = label_to_pm_one(pred.get("label", ""))

    # Zapis wynikow do pliku
    
    
    with out_path.open("w", encoding="utf-8") as f:
        for y in y_pred:
            f.write(f"{y}\n")

    # Accuracy liczymy tylko dla istniejacych plikow z etykietami
    
    
    total = 0
    correct = 0
    nolabel = 0
    missing = 0


    for i, ((_, y_t), ok) in enumerate(zip(pairs, exists_mask)): # przejscie po wszystkich parach i masce istnienia
        if not ok:
            missing += 1
            continue
        if y_t not in (-1, 1):
            nolabel += 1
            continue
        total += 1
        if y_pred[i] == y_t:
            correct += 1

    if total > 0:
        
        
        acc = correct / total
        print(f"Accuracy: {acc:.3f} ({correct}/{total})")
    else:
        print("Accuracy: nie mozemy policzyc (brak dostepnych plikow z etykietami).")





    print(f"Missing files skipped: {missing}")
    
    
    print(f"Lines without label skipped: {nolabel}")


if __name__ == "__main__":
    main()
