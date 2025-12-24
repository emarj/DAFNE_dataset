import requests
import json
from importlib import resources

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import PurePosixPath

BASE_URL = "https://vision.unipv.it/DAFchallenge/DAFNE_dataset/dataset_download.html"

def scrape_frescos() -> dict:
    response = requests.get(BASE_URL)
    response.raise_for_status()
    

    soup = BeautifulSoup(response.text, "html.parser")
    frescos = {}

    # Find all <ul> tags and their <li> items
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li"):
            a_tags = li.find_all("a", href=True)
            for a in a_tags:
                if a["href"].lower().endswith(".zip"):
                    # Convert relative links to absolute, leave absolute links unchanged
                    full_url = urljoin(BASE_URL, a["href"])
                    frescos[extract_id_from_url(full_url)] = full_url

    frescos = {k: v for k, v in frescos.items() if "DB2" not in k}
    return frescos

def retrieve_frescos(scrape=True) -> dict:
    frescos = load_metadata()
    if scrape:
        try:
            frescos = scrape_frescos()
        except requests.RequestException as e:
            print("Wraning: Failed to scrape frescos from website, falling back to local metadata.")
            print(f"(Caused by: {e})")
    
    return frescos

def extract_id_from_url(url) -> str:
    fresco_id = PurePosixPath(urlparse(url).path).stem
    return fresco_id

def load_metadata() -> dict:
    with resources.files("dafne_dataset").joinpath("frescos.json").open("r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    frescos = retrieve_frescos()
    print(f'FRESCOS = {frescos}')