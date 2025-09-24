#!/usr/bin/env python3
"""
Script to convert cluster data from JSONL format to a LaTeX table
mapping cluster IDs to cluster labels.
"""

import json
import sys
import argparse
from typing import List, Dict

def escape_latex_special_chars(text: str) -> str:
    """
    Escape special LaTeX characters in text.
    """
    replacements = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '\\': '\\textbackslash{}'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def parse_cluster_data(jsonl_content: str) -> List[Dict]:
    """
    Parse JSONL content and extract cluster data.
    """
    clusters = []
    lines = jsonl_content.strip().split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            cluster_data = json.loads(line)
            clusters.append({
                'id': cluster_data['cluster_id'],
                'label': cluster_data['label'],
                'size': cluster_data['cluster_size'],
                'score': cluster_data.get('score', 'N/A')
            })
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
            continue
        except KeyError as e:
            print(f"Warning: Missing required field {e} in line {line_num}", file=sys.stderr)
            continue
    
    return clusters

def create_latex_table(clusters: List[Dict], include_size: bool = False, include_score: bool = False) -> str:
    """
    Create a LaTeX table from cluster data.
    """
    # Sort clusters by ID
    clusters.sort(key=lambda x: x['id'])
    
    # Determine columns based on options
    columns = ['c', 'l']  # ID and Label by default
    headers = ['\\textbf{Cluster ID}', '\\textbf{Cluster Label}']
    
    if include_size:
        columns.append('c')
        headers.append('\\textbf{Size}')
    
    if include_score:
        columns.append('c')
        headers.append('\\textbf{Score}')
    
    column_spec = '|' + '|'.join(columns) + '|'
    
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Cluster ID to Label Mapping}}
\\label{{tab:cluster_labels}}
\\begin{{tabular}}{{{column_spec}}}
\\hline
{' & '.join(headers)} \\\\
\\hline
"""
    
    for cluster in clusters:
        escaped_label = escape_latex_special_chars(cluster['label'])
        
        row_data = [str(cluster['id']), escaped_label]
        
        if include_size:
            row_data.append(str(cluster['size']))
        
        if include_score:
            row_data.append(str(cluster['score']))
        
        latex_table += ' & '.join(row_data) + ' \\\\\n\\hline\n'
    
    latex_table += """\\end{tabular}
\\end{table}"""
    
    return latex_table

def main():
    """
    Main function to handle command line arguments and process the data.
    """
    parser = argparse.ArgumentParser(
        description='Convert cluster JSONL data to LaTeX table'
    )
    parser.add_argument(
        'input_file', 
        nargs='?', 
        default='-',
        help='Input JSONL file (default: stdin)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--include-size',
        action='store_true',
        help='Include cluster size column'
    )
    parser.add_argument(
        '--include-score',
        action='store_true',
        help='Include cluster score column'
    )
    
    args = parser.parse_args()
    
    # Read input
    if args.input_file == '-':
        jsonl_content = sys.stdin.read()
    else:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                jsonl_content = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.input_file}' not found", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading file '{args.input_file}': {e}", file=sys.stderr)
            sys.exit(1)
    
    # Parse data
    clusters = parse_cluster_data(jsonl_content)
    
    if not clusters:
        print("Error: No valid cluster data found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processed {len(clusters)} clusters", file=sys.stderr)
    
    # Generate LaTeX table
    latex_output = create_latex_table(
        clusters, 
        include_size=args.include_size,
        include_score=args.include_score
    )
    
    # Write output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(latex_output)
            print(f"LaTeX table written to '{args.output}'", file=sys.stderr)
        except IOError as e:
            print(f"Error writing to file '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(latex_output)

if __name__ == '__main__':
    main()