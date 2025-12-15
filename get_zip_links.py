import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://vision.unipv.it/DAFchallenge/DAFNE_dataset/dataset_download.html"

def get_zip_links(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    zip_links = []

    # Find all <ul> tags and their <li> items
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li"):
            a_tags = li.find_all("a", href=True)
            for a in a_tags:
                if a["href"].lower().endswith(".zip"):
                    # Convert relative links to absolute, leave absolute links unchanged
                    full_url = urljoin(url, a["href"])
                    zip_links.append(full_url)

    zip_links = [l for l in zip_links if "DB2" not in l]
    return zip_links


print(get_zip_links(BASE_URL))