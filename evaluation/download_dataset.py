import os
import json
import zipfile
import wget
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

# HotpotQA
# hotpot = load_dataset("hotpot_qa", "distractor")["validation"]
# with open("data/hotpot_dev.json", "w") as f:
#     for item in hotpot:
#         f.write(json.dumps(item) + "\n")

# MuSiQue
# musique = load_dataset("dgslibisey/MuSiQue", "default")["validation"]
# with open("data/musique_dev.json", "w") as f:
#     for item in musique:
#         f.write(json.dumps(item) + "\n")


DOWNLOAD_URL = "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1"
ZIP_FILENAME = "data.zip"
DATA_DIR = "data/data"
OUTPUT_FILE = os.path.join(DATA_DIR, "2wiki_dev_converted.json")

def download_and_unzip():
    if not os.path.exists(ZIP_FILENAME):
        print("📥 Downloading dataset...")
        wget.download(DOWNLOAD_URL, ZIP_FILENAME)
        print("\n✅ Download complete.")
    else:
        print("✔️ Zip file already exists.")

    print("📦 Unzipping...")
    with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("✅ Unzipped to:", DATA_DIR)

def convert_2wiki_dev():
    input_path = os.path.join(DATA_DIR, "dev.json")
    output_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔁 Converting {len(data)} entries...")

    for item in data:
        new_item = {
            "id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "context": [],
            "supporting_facts": item.get("supporting_facts", [])
        }

        for title, sentences in item["context"]:
            paragraph = " ".join(sentences)
            new_item["context"].append([title, paragraph])

        output_data.append(new_item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted and saved to {OUTPUT_FILE}")

download_and_unzip()
convert_2wiki_dev()