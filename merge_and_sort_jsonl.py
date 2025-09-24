import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def read_and_parse_lines(fname):
    records = []
    with open(fname, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                records.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Error parsing line: {e}")
    return records

def merge_and_sort_jsonl_parallel(input_file1, input_file2, output_file):
    input_files = [input_file1, input_file2]
    total_lines = 0

    # Get total line count for progress bar
    for fname in input_files:
        with open(fname, 'r', encoding='utf-8') as infile:
            for _ in infile:
                total_lines += 1

    # Parallel read and parse
    with ThreadPoolExecutor() as executor:
        # Submit all files to be read/parsed in parallel
        future_to_file = {executor.submit(read_and_parse_lines, fname): fname for fname in input_files}
        records = []
        with tqdm(total=total_lines, desc="Reading input files", unit="lines") as pbar:
            for future in future_to_file:
                file_records = future.result()
                records.extend(file_records)
                pbar.update(len(file_records))

    # Sort
    records.sort(key=lambda x: (x['paper_id'], x['section_index'], x['paragraph_index']))

    # Write out
    with tqdm(total=len(records), desc="Writing output file", unit="lines") as pbar:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for rec in records:
                outfile.write(json.dumps(rec) + '\n')
                pbar.update(1)

# Usage
input_file1 = 'data/unlabelled_methodology/predicted_paragraphs.jsonl'
input_file2 = 'data/methodolodgy_labels/labeled_paragraphs.jsonl'
output_file = 'data/methodology_dataset/merged_sorted.jsonl'

merge_and_sort_jsonl_parallel(input_file1, input_file2, output_file)
