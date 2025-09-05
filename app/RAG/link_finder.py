import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fastapi import HTTPException


def get_internal_links(base_url: str):
    """Fetch and return all unique internal links from the given website."""
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {base_url}: {str(e)}")

    soup = BeautifulSoup(response.text, "html.parser")
    base_domain = urlparse(base_url).netloc

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc == base_domain:  # keep only internal links
            links.add(parsed.path)

    return sorted(list(links))