import os

root_dir = "."  # Change si besoin
target = "AutoencoderKLHunyuanVideo"

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".py"):
            path = os.path.join(dirpath, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if target in line:
                        print(f"{path}:{i}: {line.strip()}")
