import re
import json
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# -----------------------
# CONFIG
# -----------------------
BASE_LISTING_URL = "https://www.squarepharma.com.bd/products-by-tradename.php"
PRODUCT_URL_RE = re.compile(r"product-details\.php\?pid=\d+", re.IGNORECASE)

CHAR_BUCKETS = [chr(c) for c in range(ord("A"), ord("B") + 1)]

# ✅ Run BOTH "pharma" and "herbal" in one execution
RUN_TYPES = ["pharma", "herbal"]  # (site types: pharma | herbal | agrovet)

ROOT_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT_DIR / "data"
JSON_DIR = ROOT_DIR / "save"
OUT_JSON = JSON_DIR / "product_pdf_map.json"

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.6
UA = "Mozilla/5.0 (compatible; SquarePharmaPDFDownloader/1.2)"


# -----------------------
# HELPERS
# -----------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def listing_url(prod_type: str, ch: str) -> str:
    return f"{BASE_LISTING_URL}?type={prod_type}&char={ch}"

def sanitize_windows_filename(name: str, max_len: int = 140) -> str:
    name = clean_text(name)
    name = unicodedata.normalize("NFKC", name)

    # remove common symbols from filenames
    name = name.replace("®", "").replace("™", "").replace("©", "").replace("TM", "")
    name = clean_text(name)

    # Windows forbidden chars: \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]+', "-", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"-{2,}", "-", name).strip()
    name = name.rstrip(" .")

    if not name:
        name = "unknown"

    if len(name) > max_len:
        name = name[:max_len].rstrip(" .-")

    return name

def unique_path(path: Path) -> Path:
    """If path exists, add (1), (2), ... to avoid overwriting."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand
        i += 1

def extract_product_name(soup: BeautifulSoup) -> str:
    h1 = soup.select_one("#toptizerpdetails h1.pdetails")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))
    h1_any = soup.find("h1")
    if h1_any:
        return clean_text(h1_any.get_text(" ", strip=True))
    return ""

def extract_prescribing_pdf_url(page_url: str, soup: BeautifulSoup) -> str:
    # Prefer the “View Prescribing Details” link
    a = soup.find("a", string=lambda t: t and "Prescribing" in t)
    if a and a.get("href"):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            return urljoin(page_url, href)

    # Fallback: any PDF inside downloads/
    for a in soup.select('a[href*="downloads/"]'):
        href = (a.get("href") or "").strip()
        if href.lower().endswith(".pdf"):
            return urljoin(page_url, href)

    return ""

def extract_product_links(listing_page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if href and PRODUCT_URL_RE.search(href):
            links.add(urljoin(listing_page_url, href))
    return sorted(links)

def pdf_basename_from_url(pdf_url: str) -> str:
    p = urlparse(pdf_url)
    name = Path(unquote(p.path)).name
    return name or "unknown.pdf"

def load_state() -> dict:
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"items": [], "failures": []}
            data.setdefault("items", [])
            data.setdefault("failures", [])
            return data
        except Exception:
            return {"items": [], "failures": []}
    return {"items": [], "failures": []}

def save_state(state: dict) -> None:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    state["count"] = len(state.get("items", []))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# -----------------------
# NETWORK
# -----------------------
class Client:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})

    def sleep(self):
        time.sleep(SLEEP_SECONDS)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        retry=retry_if_exception_type((requests.RequestException,)),
        reraise=True,
    )
    def fetch_html(self, url: str) -> str:
        r = self.s.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        retry=retry_if_exception_type((requests.RequestException,)),
        reraise=True,
    )
    def download_file(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.s.get(url, timeout=REQUEST_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)


# -----------------------
# MAIN
# -----------------------
def dataCrawler(run_types: list[str] = RUN_TYPES):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    c = Client()
    state = load_state()

    items = state.get("items", [])
    failures = state.get("failures", [])

    # ✅ Uniqueness is GLOBAL across pharma+herbal (based on original PDF filename)
    seen_original_pdfs = set()
    for it in items:
        op = (it.get("original_pdf_filename") or "").strip()
        if op:
            seen_original_pdfs.add(op.lower())

    print(f"Loaded {len(items)} existing records.")
    print(f"Seen original PDF filenames: {len(seen_original_pdfs)}")
    print(f"Run types: {run_types}")

    new_downloads = 0
    total_pages = 0

    for prod_type in run_types:
        # 1) Collect product URLs A-Z for this type
        all_product_urls = set()
        for ch in CHAR_BUCKETS:
            url = listing_url(prod_type, ch)
            try:
                html = c.fetch_html(url)
                urls = extract_product_links(url, html)
                all_product_urls.update(urls)
                print(f"[{prod_type.upper()} {ch}] products: {len(urls)}")
            except Exception as e:
                failures.append({"stage": "listing", "type": prod_type, "url": url, "error": str(e)})
                print(f"[{prod_type.upper()} {ch}] listing failed: {e}")
            finally:
                c.sleep()

        all_product_urls = sorted(all_product_urls)
        total_pages += len(all_product_urls)
        print(f"\n[{prod_type.upper()}] Total product pages collected: {len(all_product_urls)}")

        # 2) Visit each product page; download only if original_pdf_filename not already in JSON
        for i, purl in enumerate(all_product_urls, start=1):
            try:
                html = c.fetch_html(purl)
                soup = BeautifulSoup(html, "lxml")

                product_name = extract_product_name(soup)
                if not product_name:
                    raise ValueError("Missing product name (h1.pdetails)")

                pdf_url = extract_prescribing_pdf_url(purl, soup)
                if not pdf_url:
                    continue

                original_pdf_filename = pdf_basename_from_url(pdf_url)
                if not original_pdf_filename:
                    continue

                key = original_pdf_filename.lower()
                if key in seen_original_pdfs:
                    print(f"[{prod_type} {i}/{len(all_product_urls)}] SKIP (already in JSON): {product_name} -> {original_pdf_filename}")
                    continue

                # rename BEFORE download: "<product>.pdf"
                safe_product = sanitize_windows_filename(product_name)
                desired_name = f"{safe_product}.pdf"
                out_path = unique_path(PDF_DIR / desired_name)

                record = {
                    "type": prod_type,  # ✅ distinguish pharma vs herbal
                    "product_name": product_name,
                    "product_url": purl,
                    "pdf_url": pdf_url,
                    "original_pdf_filename": original_pdf_filename,
                    "renamed_pdf_filename": out_path.name,
                    "downloaded": False,
                }

                # Save JSON BEFORE download
                items.append(record)
                seen_original_pdfs.add(key)
                state["items"] = items
                state["failures"] = failures
                save_state(state)

                # Download
                c.download_file(pdf_url, out_path)
                record["downloaded"] = True
                save_state(state)

                new_downloads += 1
                print(f"[{prod_type} {i}/{len(all_product_urls)}] DOWNLOADED: {product_name} -> {out_path.name} (orig: {original_pdf_filename})")

            except Exception as e:
                failures.append({"stage": "product", "type": prod_type, "url": purl, "error": str(e)})
                state["failures"] = failures
                save_state(state)
                print(f"[{prod_type} {i}/{len(all_product_urls)}] product failed: {purl} :: {e}")
            finally:
                c.sleep()

    state["items"] = items
    state["failures"] = failures
    save_state(state)

    print(f"\nSaved JSON → {OUT_JSON}")
    print(f"PDF dir → {PDF_DIR}")
    print(f"Total pages visited (pharma+herbal) → {total_pages}")
    print(f"New unique PDFs downloaded this run → {new_downloads}")
    print(f"Total failures recorded → {len(failures)}")
    print("Done.")


if __name__ == "__main__":
    dataCrawler()
