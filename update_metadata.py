import json

from dafne_dataset.metadata import retrieve_frescos

def write_metadata(metadata: dict):
    with open("dafne_dataset/frescos.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    frescos = retrieve_frescos()
    write_metadata(frescos)