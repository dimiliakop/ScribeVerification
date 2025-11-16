from pathlib import Path
from PIL import Image

def check_images(root="./data"):
    root = Path(root)
    bad_files = []
    for p in root.rglob("*.*"):
        if p.suffix.lower() not in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
            continue
        try:
            with Image.open(p) as img:
                img.verify()  # check header
        except Exception as e:
            print(f"[BAD] {p} ({e})")
            bad_files.append(str(p))
    print(f"Checked {len(list(root.rglob('*.*')))} files, found {len(bad_files)} corrupted.")
    return bad_files

if __name__ == "__main__":
    check_images("./data")
