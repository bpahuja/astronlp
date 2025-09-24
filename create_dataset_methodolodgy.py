import os
import json
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re
from collections import Counter
import html

# CONFIGURATION
PREFIX = "/vol/bitbucket/astronlp/data/arXivUpdate/"
MANIFEST_FILE = "data/unlabelled_methodology/manifest_sample_unlabelled.json"        # JSON with {paper_key: path}
OUTPUT_FILE = "data/unlabelled_methodology/unlabelled_paragraphs.jsonl"
N_WORKERS = 16
MIN_PARAGRAPH_WORDS = 25

# Setup logging
logging.basicConfig(
    filename='data/methodolodgy_labels/labeling.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# HEURISTIC RULES
METHOD_HEADINGS = [
    "method", "approach", "observations", "data reduction", "instrumentation",
    "analysis", "procedure", "data processing", "experimental setup",
    "technique", "telescope configuration", "modelling", "measurement", "pipeline"
]


METHOD_REGEXES = [
    re.compile(r"\bwe observed\b", re.IGNORECASE),
    re.compile(r"\bdata (were|was) collected\b", re.IGNORECASE),
    re.compile(r"\bspectra (were|was) reduced\b", re.IGNORECASE),
    # re.compile(r"\bphotometric measurements\b", re.IGNORECASE),
    # re.compile(r"\b(telescope|instrument)(s)? used\b", re.IGNORECASE),
    # re.compile(r"\baperture photometry\b", re.IGNORECASE),
    # re.compile(r"\blight curves? (were|was)? extracted\b", re.IGNORECASE),
    re.compile(r"\bmodel(l)?ed\b", re.IGNORECASE),
    re.compile(r"\bwe derived\b", re.IGNORECASE),
    re.compile(r"\bdata analysis (was|were)? performed\b", re.IGNORECASE),
    re.compile(r"\bexposure time\b", re.IGNORECASE),
    re.compile(r"\buncertainty estimation\b", re.IGNORECASE),
    # re.compile(r"\bconfiguration of the telescope\b", re.IGNORECASE)
]

def is_methodology_paragraph(section_heading: str, paragraph_text: str) -> bool:
    heading = (section_heading or "").lower()
    paragraph = paragraph_text.lower()

    # Strong match on section heading
    if any(keyword in heading for keyword in METHOD_HEADINGS):
        return True

    # Regex-based phrase matching
    for pattern in METHOD_REGEXES:
        if pattern.search(paragraph):
            return True

    return False


def get_full_text(element):
    """
    Recursively extract text, replacing <Math> with tokenizer-friendly placeholders.
    Example: <Math tex="E=mc^2">E = mc^2</Math> → [MATH_tex=E=mc^2]
    """
    if element is None:
        return ""

    parts = []

    def recurse(elem):

        # Handle <Math> tag
        if elem.tag.lower() == "math":
            tex = elem.attrib.get("tex", "").strip()
            if tex:
                # Escape brackets and backslashes to avoid confusing the tokenizer
                clean_tex = tex.replace("\\", "").replace("{", "").replace("}", "")
                parts.append(f"[MATH_tex={clean_tex}]")
        # Add element text before children
        elif elem.text:
            parts.append(elem.text.strip())
        
        # Recurse into children
        for child in elem:
            recurse(child)

        # Add tail text after children
        if elem.tail:
            parts.append(elem.tail.strip())

    recurse(element)
    return " ".join(filter(None, parts)).strip()




# XML PARSER PER FILE
def extract_paragraphs(paper_key_path):
    """Parse a LaTeXML file, extract paragraphs, and label them."""
    paper_key, xml_path = paper_key_path
    paragraphs = []

    try:
        tree = ET.parse(PREFIX + xml_path)
        root = tree.getroot()

        # Loop through each section
        for section_idx, section in enumerate(root.iter("section")):
            title_elem = section.find("title")
            section_title = get_full_text(title_elem)
            para_list = section.findall("para")

            for para_idx, para in enumerate(para_list):
                para_text_elements = para.findall("p")
                paragraph_text = " ".join(
                    get_full_text(elem) for elem in para_text_elements
                )

                if len(paragraph_text.split()) < MIN_PARAGRAPH_WORDS:
                    continue

                label = "methodology" if is_methodology_paragraph(section_title, paragraph_text) else "not_methodology"

                paragraphs.append({
                    "paper_id": paper_key,
                    "section": section_title,
                    "paragraph": paragraph_text,
                    "label": label,
                    "section_index": section_idx,
                    "paragraph_index": para_idx
                })

        logging.info(f"Processed: {paper_key} ({len(paragraphs)} paragraphs)")

    except ET.ParseError as e:
        logging.warning(f"XML ParseError in {paper_key}: {e}")
    except Exception as e:
        logging.error(f"Failed processing {paper_key}: {e}")

    return paragraphs


# MAIN PARALLEL RUN
def process_all_from_manifest(manifest_path, output_file, n_workers):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    key_path_pairs = list(manifest.items())
    total_files = len(key_path_pairs)
    logging.info(f"Loaded {total_files} entries from manifest")

    with open(output_file, "w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(extract_paragraphs, item): item[0]
                for item in key_path_pairs
            }

            for future in tqdm(as_completed(futures), total=total_files, desc="Processing papers"):
                try:
                    result = future.result()
                    for para in result:
                        out_f.write(json.dumps(para) + "\n")
                except Exception as e:
                    key = futures[future]
                    logging.error(f"Unhandled error in {key}: {e}")

    logging.info("All processing complete.")


def summarize_stats(jsonl_path):
    counts = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            counts[entry["label"]] += 1

    total = counts["methodology"] + counts["not_methodology"]
    print("\n📊 Label Distribution Summary")
    print("------------------------------")
    print(f"Total paragraphs       : {total}")
    print(f"Methodology paragraphs : {counts['methodology']}")
    print(f"Other paragraphs       : {counts['not_methodology']}")
    print(f"Methodology %          : {counts['methodology'] / total:.2%}")
    print(f"Other %                : {counts['not_methodology'] / total:.2%}")


# ENTRY POINT
if __name__ == "__main__":
    process_all_from_manifest(MANIFEST_FILE, OUTPUT_FILE, N_WORKERS)
    summarize_stats(OUTPUT_FILE)
