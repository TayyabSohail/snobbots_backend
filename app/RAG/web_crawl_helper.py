import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def get_html_with_fallback(url: str, timeout: int = 30) -> str:
    """Try normal GET first; if page looks empty, render JS with requests_html."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36"
    }

    try:
        # Try static fetch first (fast)
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()

        # If HTML is short or contains almost no text → probably JS rendered
        if len(r.text) < 1000 or "<script" in r.text and "<body" in r.text and not any(t in r.text for t in ["<p", "<div", "<span"]):
            session = HTMLSession()
            r = session.get(url, headers=headers)
            r.html.render(timeout=timeout, sleep=5)
            html = r.html.html
            session.close()
            return html

        return r.text

    except Exception as e:
        raise Exception(f"Failed to fetch or render {url}: {e}")

def get_internal_links_js(url: str):
    html = get_html_with_fallback(url)
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(url).netloc

    links = set()
    for a in soup.find_all("a", href=True):
        href = a['href']
        full_url = urljoin(url, href)
        if urlparse(full_url).netloc == base_domain:
            links.add(full_url)

    return list(links)