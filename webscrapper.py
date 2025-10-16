import os
import re
import requests
from bs4 import BeautifulSoup

urls = [
    "https://canonical.com/solutions/ai",
]

headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def sanitize_filename(name: str) -> str:
    """Remove invalid characters and format for saving."""
    name = re.sub(r"[\\/*?\"<>|:]", "", name)
    name = name.replace(" ", "_")
    return name.strip() or "untitled_page"

root_data_path = "web_pages"

for url in urls:
    print(f"Fetching {url}...")
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Get title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled Page"
    safe_title = sanitize_filename(title)
    filename = f"{root_data_path}/{safe_title}.txt"

    # Find main content
    soup_main = soup.find("div", id="main-content")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Title: {title}\n\n")

        if not soup_main:
            f.write("No main-content div found.\n")
            print(f"⚠️  No main-content found for {url}")
            continue

        sections = soup_main.find_all("section", class_="p-section")
        if not sections:
            f.write("No p-section sections found.\n")
            print(f"⚠️  No p-section found for {url}")
            continue

        # Write all text sections
        for idx, section in enumerate(sections, start=1):
            text = section.get_text(" ", strip=True)
            f.write(f"--- Section {idx} ---\n{text}\n\n")

    print(f"✅ Saved: {filename}")
