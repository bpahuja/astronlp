import os
import json
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from keybert import KeyBERT

# CONFIGURATION
PREFIX = "/vol/bitbucket/astronlp/data/arXivUpdate/"
MANIFEST_FILE = "/vol/bitbucket/astronlp/data/arXivUpdate/manifest.json"
OUTPUT_FILE = "/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/titles_abstracts_keywords.jsonl"
N_WORKERS = 16
NUM_KEYWORDS = 10

# Setup logging
logging.basicConfig(
    filename='/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/title_abstract_extraction.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_full_text(element):
    """Extract all text from an XML element, including subelements."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()

def extract_keywords_bert(text, num_keywords=10):
    if not hasattr(extract_keywords_bert, "kw_model"):
        bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        extract_keywords_bert.kw_model = KeyBERT(model=bert_model)
    kw_model = extract_keywords_bert.kw_model
    if not text or not text.strip():
        return []
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=num_keywords
    )
    return [kw[0] for kw in keywords]

def extract_title_abstract_keywords(paper_key_path):
    """Extract title, abstract, and keywords from a LaTeXML file."""
    paper_key, xml_path = paper_key_path
    result = {
        "paper_id": paper_key,
        "title": None,
        "abstract": None,
        "keywords": [],
    }
    try:
        tree = ET.parse(PREFIX + xml_path)
        root = tree.getroot()

        # Try to find title and abstract at the root or as direct children
        title_elem = root.find("title")
        if title_elem is None:
            for child in root:
                if child.tag.lower() == "title":
                    title_elem = child
                    break
        result["title"] = get_full_text(title_elem) if title_elem is not None else None

        abstract_elem = root.find("abstract")
        if abstract_elem is None:
            for child in root:
                if child.tag.lower() == "abstract":
                    abstract_elem = child
                    break
        abstract_text = get_full_text(abstract_elem) if abstract_elem is not None else None
        result["abstract"] = abstract_text

        # Extract keywords from abstract
        if abstract_text:
            result["keywords"] = extract_keywords_bert(abstract_text, NUM_KEYWORDS)
        else:
            result["keywords"] = []

        result["xml_path"] = xml_path  # For debugging purposes

        logging.info(f"Extracted title/abstract/keywords for: {paper_key}")

    except ET.ParseError as e:
        logging.warning(f"XML ParseError in {paper_key}: {e}")
        print(f"XML ParseError in {paper_key}: {e}")
    except Exception as e:
        logging.error(f"Failed processing {paper_key}: {e}")
        print(f"Failed processing {paper_key}: {e}")

    return result

def process_all_titles_abstracts_keywords(manifest_path, output_file, n_workers):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    key_path_pairs = list(manifest.items())
    total_files = len(key_path_pairs)
    logging.info(f"Loaded {total_files} entries from manifest")

    with open(output_file, "w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(extract_title_abstract_keywords, item): item[0]
                for item in key_path_pairs
            }
            for future in tqdm(as_completed(futures), total=total_files, desc="Extracting titles/abstracts/keywords"):
                try:
                    result = future.result()
                    out_f.write(json.dumps(result) + "\n")
                except Exception as e:
                    key = futures[future]
                    logging.error(f"Unhandled error in {key}: {e}")

    logging.info("All title/abstract/keyword extraction complete.")

if __name__ == "__main__":
    process_all_titles_abstracts_keywords(MANIFEST_FILE, OUTPUT_FILE, N_WORKERS)