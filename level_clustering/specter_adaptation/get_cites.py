import os
import json
import logging
import xml.etree.ElementTree as ET
from tqdm import tqdm
import asyncio
import aiohttp

# CONFIGURATION
PREFIX = "/vol/bitbucket/astronlp/data/arXivUpdate/"
MANIFEST_FILE = "/vol/bitbucket/astronlp/data/arXivUpdate/manifest.json"
OUTPUT_FILE = "/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/citations.jsonl"
N_WORKERS = 16
ARXIV_RATE_LIMIT = 3  # requests per second

# Setup logging
logging.basicConfig(
    filename='/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/citation_extraction.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_references(xml_path):
    refs = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for bibitem in root.findall(".//bibitem"):
            title = bibitem.findtext(".//text[@font='italic']")
            authors = bibitem.findtext(".//bibblock")
            refs.append({'title': title, 'authors': authors})
    except Exception as e:
        logging.warning(f"Failed to parse references in {xml_path}: {e}")
    return refs

async def search_arxiv(session, title, semaphore):
    url = f'http://export.arxiv.org/api/query?search_query=ti:"{title}"&max_results=1'
    async with semaphore:
        await asyncio.sleep(1 / ARXIV_RATE_LIMIT)  # crude rate limiting
        try:
            async with session.get(url, timeout=10) as resp:
                text = await resp.text()
                if resp.status == 200 and '<id>http://arxiv.org/abs/' in text:
                    start = text.find('<id>http://arxiv.org/abs/') + len('<id>http://arxiv.org/abs/')
                    end = text.find('</id>', start)
                    return text[start:end]
        except Exception as e:
            logging.warning(f"arXiv API error for title '{title}': {e}")
    return None

async def process_paper(paper_key, xml_path, session, semaphore):
    abs_path = os.path.join(PREFIX, xml_path)
    refs = extract_references(abs_path)
    citations = []
    for ref in refs:
        title = ref.get('title')
        if title:
            arxiv_id = await search_arxiv(session, title, semaphore)
            if arxiv_id:
                citations.append({
                    'title': title,
                    'authors': ref.get('authors'),
                    'arxiv_id': arxiv_id
                })
    # Ignore if no citations with arxiv_id
    if not citations:
        return None
    return {
        "paper_id": paper_key,
        "citations": citations
    }

async def main():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    key_path_pairs = list(manifest.items())
    total_files = len(key_path_pairs)

    semaphore = asyncio.Semaphore(ARXIV_RATE_LIMIT)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_paper(paper_key, xml_path, session, semaphore)
            for paper_key, xml_path in key_path_pairs
        ]
        for f in tqdm(asyncio.as_completed(tasks), total=total_files, desc="Extracting citations"):
            result = await f
            if result and result["paper_id"]:
                results.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for entry in results:
            out_f.write(json.dumps(entry) + "\n")

    logging.info("Citation extraction complete.")

if __name__ == "__main__":
    asyncio.run(main())