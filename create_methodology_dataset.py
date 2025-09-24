import os
import json
from tqdm import tqdm

def write_paper(paper_id, paragraphs, output_dir):
    sorted_paras = sorted(paragraphs, key=lambda x: (x['section_index'], x['paragraph_index']))
    text = ' '.join(p['paragraph'] for p in sorted_paras)
    paper_id = paper_id.replace('/', '_')
    txt_path = os.path.join(output_dir, f"{paper_id}.txt")
    if os.path.exists(txt_path):
        # Optionally, still return meta if you want metadata updated
        return paper_id, [
            {"section_index": p['section_index'], "paragraph_index": p['paragraph_index']}
            for p in sorted_paras
        ]

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    meta = [
        {"section_index": p['section_index'], "paragraph_index": p['paragraph_index']}
        for p in sorted_paras
    ]
    return paper_id, meta

def stream_and_process_jsonl(input_path, output_dir, metadata_path):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}
    current_paper_id = None
    current_paragraphs = []
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))

    # First pass: Group lines by paper_id
    with open(input_path, 'r', encoding='utf-8') as infile, \
         tqdm(total=total_lines, desc='Processing', unit='lines') as pbar:

        for line in infile:
            pbar.update(1)
            entry = json.loads(line.strip())
            label = entry.get('label')
            predicted_label = entry.get('predicted_label')

            # Only keep relevant paragraphs
            if predicted_label is None and label not in ['methodology', 1]:
                continue
            if predicted_label is not None and predicted_label not in ['methodology', 1]:
                continue

            paper_id = entry['paper_id']
            if current_paper_id is not None and paper_id != current_paper_id:
                # Sequentially write out the last paper's data
                pid, meta = write_paper(current_paper_id, current_paragraphs, output_dir)
                metadata[pid] = meta
                current_paragraphs = []
            current_paper_id = paper_id
            current_paragraphs.append(entry)

        # Flush the last group
        if current_paper_id and current_paragraphs:
            pid, meta = write_paper(current_paper_id, current_paragraphs, output_dir)
            metadata[pid] = meta

    # Save metadata after all writing is done
    with open(metadata_path, 'w', encoding='utf-8') as meta_f:
        json.dump(metadata, meta_f, indent=2)

# Usage
input_jsonl = 'data/methodology/merged_sorted.jsonl'
output_dir = 'data/methodology_dataset'
metadata_path = os.path.join(output_dir, 'metadata.json')
stream_and_process_jsonl(input_jsonl, output_dir, metadata_path)
