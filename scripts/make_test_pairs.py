import csv, random, os
from pathlib import Path

def collect(root):
    root = Path(root)
    m = {}
    for cls in sorted([d for d in root.iterdir() if d.is_dir()]):
        imgs = sorted([str(p) for p in cls.glob("*") if p.suffix.lower() in
                      [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]])
        if imgs:
            m[cls.name] = imgs
    return m

if __name__ == "__main__":
    root = Path("./data/test")
    out_csv = Path("./data/test_pairs.csv")
    class_to_imgs = collect(root)
    classes = list(class_to_imgs.keys())

    pairs = []
    target_pairs = 30000   # limit to ~5k pairs (balanced pos/neg)

    while len(pairs) < target_pairs:
        # positive pair
        c = random.choice(classes)
        if len(class_to_imgs[c]) >= 2:
            a, b = random.sample(class_to_imgs[c], 2)
            pairs.append((a, b, 1))
        # negative pair
        c1, c2 = random.sample(classes, 2)
        a = random.choice(class_to_imgs[c1])
        b = random.choice(class_to_imgs[c2])
        pairs.append((a, b, 0))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path1", "path2", "label"])
        for p in pairs:
            w.writerow(p)

    print(f"Saved {len(pairs)} pairs to {out_csv}")
