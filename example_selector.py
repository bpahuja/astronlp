"""
Text Summary Example Selector for Academic Reports

This script selects representative text-summary pairs from your analysis results
for inclusion in thesis reports, papers, or presentations.

Usage examples:
    # Select diverse examples with full original text (no truncation)
    python example_selector.py \
        --data_file results/bart_analysis_data.csv \
        --original_dir data/original_texts \
        --summary_dir data/bart_summaries \
        --num_examples 6 \
        --selection_strategy quality_range \
        --output_format latex \
        --output_file thesis_examples.tex \
        --full_original_text

    # Select examples with different compression ratios (if text is in CSV)
    python example_selector.py \
        --data_file results/model_data.csv \
        --selection_strategy compression_range \
        --output_format markdown \
        --max_summary_length 200 \
        --full_original_text

    # Select specific examples by ID with full text display
    python example_selector.py \
        --data_file results/data.csv \
        --original_dir data/originals \
        --summary_dir data/summaries \
        --specific_ids doc1,doc5,doc12 \
        --output_format latex \
        --max_summary_length 500 \
        --full_original_text
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import textwrap
import re

class ExampleSelector:
    def __init__(self, data_file: str, original_dir: str = None, summary_dir: str = None):
        """
        Initialize the example selector.
        
        Args:
            data_file: Path to analysis data CSV file
            original_dir: Directory containing original texts (optional if texts are in CSV)
            summary_dir: Directory containing summaries (optional if texts are in CSV)
        """
        self.data_file = Path(data_file)
        self.original_dir = Path(original_dir) if original_dir else None
        self.summary_dir = Path(summary_dir) if summary_dir else None
        self.df = None
        self.selected_examples = []
        
    def load_data(self):
        """Load the analysis data."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        print(f"Loaded {len(self.df)} examples from {self.data_file}")
        
        # Check if text content is already in CSV
        has_original = 'original_text' in self.df.columns and not self.df['original_text'].isna().all()
        has_summary = 'summary_text' in self.df.columns and not self.df['summary_text'].isna().all()
        
        if has_original and has_summary:
            print("Text content found in CSV file")
            # Clean up any NaN values
            self.df['original_text'] = self.df['original_text'].fillna('')
            self.df['summary_text'] = self.df['summary_text'].fillna('')
        elif self.original_dir and self.summary_dir:
            print("Loading text content from directories...")
            self._load_text_content()
        else:
            print("Warning: Text content not available.")
            print("Provide either CSV with text columns or --original_dir and --summary_dir")
            print("Available columns:", list(self.df.columns))
    
    def _load_text_content(self):
        """Load original text and summary content from files."""
        original_texts = []
        summary_texts = []
        successful_loads = 0
        
        print("Loading text files...")
        for idx, row in self.df.iterrows():
            file_id = str(row['file_id'])
            
            # Try to find and load original text
            original_text = ""
            original_found = False
            for ext in ['', '.txt', '.text', '.dat']:
                for original_file in [
                    self.original_dir / f"{file_id}{ext}",
                    self.original_dir / f"{file_id}.original{ext}",
                    self.original_dir / f"{file_id}_original{ext}"
                ]:
                    if original_file.exists():
                        try:
                            with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
                                original_text = f.read().strip()
                            original_found = True
                            break
                        except Exception as e:
                            print(f"Error reading {original_file}: {e}")
                            continue
                if original_found:
                    break
            
            # Try to find and load summary text
            summary_text = ""
            summary_found = False
            for ext in ['', '.txt', '.text', '.dat']:
                for summary_file in [
                    self.summary_dir / f"{file_id}{ext}",
                    self.summary_dir / f"{file_id}.summary{ext}",
                    self.summary_dir / f"{file_id}_summary{ext}"
                ]:
                    if summary_file.exists():
                        try:
                            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                                summary_text = f.read().strip()
                            summary_found = True
                            break
                        except Exception as e:
                            print(f"Error reading {summary_file}: {e}")
                            continue
                if summary_found:
                    break
            
            if original_found and summary_found:
                successful_loads += 1
            elif idx < 5:  # Show first few failures for debugging
                print(f"Could not load text for {file_id}: original={original_found}, summary={summary_found}")
            
            original_texts.append(original_text)
            summary_texts.append(summary_text)
        
        self.df['original_text'] = original_texts
        self.df['summary_text'] = summary_texts
        print(f"Successfully loaded text content for {successful_loads}/{len(self.df)} examples")
        
        if successful_loads == 0:
            print("Warning: No text files could be loaded. Check your directory paths and file naming.")
            print(f"Looking in:")
            print(f"  Original: {self.original_dir}")
            print(f"  Summary: {self.summary_dir}")
            if self.original_dir and self.original_dir.exists():
                sample_files = list(self.original_dir.iterdir())[:5]
                print(f"  Sample files in original dir: {[f.name for f in sample_files]}")
            if self.summary_dir and self.summary_dir.exists():
                sample_files = list(self.summary_dir.iterdir())[:5]
                print(f"  Sample files in summary dir: {[f.name for f in sample_files]}")
    
    def select_examples(self, num_examples: int = 6, 
                       strategy: str = "quality_range", 
                       specific_ids: List[str] = None,
                       min_length: int = 50,
                       max_length: int = 500,
                       score_column: str = "rouge_l") -> List[Dict]:
        """
        Select examples based on specified strategy.
        
        Args:
            num_examples: Number of examples to select
            strategy: Selection strategy ('quality_range', 'compression_range', 'length_range', 'random')
            specific_ids: List of specific file IDs to select
            min_length: Minimum original text length for selection
            max_length: Maximum original text length for selection
            score_column: Column to use for quality-based selection
        
        Returns:
            List of selected example dictionaries
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Filter by length if specified
        filtered_df = self.df.copy()
        if min_length > 0:
            filtered_df = filtered_df[filtered_df['original_length'] >= min_length]
        if max_length > 0:
            filtered_df = filtered_df[filtered_df['original_length'] <= max_length]
        
        # Also filter to only include examples with text content
        has_text = (
            (filtered_df['original_text'].notna()) & 
            (filtered_df['summary_text'].notna()) &
            (filtered_df['original_text'].str.len() > 10) &
            (filtered_df['summary_text'].str.len() > 5)
        )
        
        text_available = filtered_df[has_text]
        if len(text_available) < len(filtered_df):
            print(f"Filtered to {len(text_available)} examples with available text content (from {len(filtered_df)} in length range)")
            filtered_df = text_available
        else:
            print(f"Filtered to {len(filtered_df)} examples within length range [{min_length}, {max_length}]")
        
        if len(filtered_df) == 0:
            print("Error: No examples available with the specified criteria and text content")
            return []
        
        if specific_ids:
            # Select specific examples by ID
            selected_df = filtered_df[filtered_df['file_id'].isin(specific_ids)]
            print(f"Selected {len(selected_df)} specific examples")
        else:
            # Select based on strategy
            selected_df = self._apply_selection_strategy(filtered_df, num_examples, strategy, score_column)
        
        # Convert to list of dictionaries
        self.selected_examples = []
        for _, row in selected_df.iterrows():
            example = {
                'file_id': str(row['file_id']),
                'original_length': int(row['original_length']),
                'summary_length': int(row['summary_length']),
                'compression': float(row['compression'])
            }
            
            # Add scores if available
            if score_column in row and pd.notna(row[score_column]):
                example[score_column] = float(row[score_column])
            if 'bertscore_recall' in row and pd.notna(row['bertscore_recall']):
                example['bertscore_recall'] = float(row['bertscore_recall'])
            
            # Add text content - ensure we have actual text
            original_text = str(row['original_text']) if pd.notna(row['original_text']) else ""
            summary_text = str(row['summary_text']) if pd.notna(row['summary_text']) else ""
            
            if original_text.strip() and summary_text.strip():
                example['original_text'] = original_text.strip()
                example['summary_text'] = summary_text.strip()
                # Calculate actual word counts for verification
                example['actual_original_words'] = len(original_text.split())
                example['actual_summary_words'] = len(summary_text.split())
            else:
                print(f"Warning: Empty text content for {example['file_id']}")
            
            self.selected_examples.append(example)
        
        return self.selected_examples
    
    def _apply_selection_strategy(self, df: pd.DataFrame, num_examples: int, 
                                 strategy: str, score_column: str) -> pd.DataFrame:
        """Apply the specified selection strategy."""
        
        if strategy == "quality_range":
            return self._select_quality_range(df, num_examples, score_column)
        elif strategy == "compression_range":
            return self._select_compression_range(df, num_examples)
        elif strategy == "length_range":
            return self._select_length_range(df, num_examples)
        elif strategy == "random":
            return df.sample(n=min(num_examples, len(df)), random_state=42)
        elif strategy == "best":
            if score_column not in df.columns:
                print(f"Warning: {score_column} not available, using compression ratio")
                return df.nlargest(num_examples, 'compression')
            return df.nlargest(num_examples, score_column)
        elif strategy == "worst":
            if score_column not in df.columns:
                print(f"Warning: {score_column} not available, using compression ratio")
                return df.nsmallest(num_examples, 'compression')
            return df.nsmallest(num_examples, score_column)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_quality_range(self, df: pd.DataFrame, num_examples: int, score_column: str) -> pd.DataFrame:
        """Select examples across quality range (best, worst, middle)."""
        if score_column not in df.columns or df[score_column].isna().all():
            print(f"Warning: {score_column} not available, using compression ratio instead")
            score_column = 'compression'
        
        df_clean = df.dropna(subset=[score_column])
        if len(df_clean) == 0:
            return df.sample(n=min(num_examples, len(df)), random_state=42)
        
        # Divide into quality ranges
        examples_per_range = max(1, num_examples // 3)
        selected_dfs = []
        
        # Best examples
        best_df = df_clean.nlargest(examples_per_range, score_column)
        selected_dfs.append(best_df)
        
        # Worst examples
        worst_df = df_clean.nsmallest(examples_per_range, score_column)
        selected_dfs.append(worst_df)
        
        # Middle examples (if we need more)
        remaining = num_examples - len(best_df) - len(worst_df)
        if remaining > 0:
            # Remove already selected examples
            remaining_df = df_clean[~df_clean.index.isin(best_df.index.union(worst_df.index))]
            if len(remaining_df) > 0:
                median_score = remaining_df[score_column].median()
                # Select examples closest to median
                remaining_df['distance_to_median'] = abs(remaining_df[score_column] - median_score)
                middle_df = remaining_df.nsmallest(remaining, 'distance_to_median')
                selected_dfs.append(middle_df)
        
        return pd.concat(selected_dfs).drop_duplicates()
    
    def _select_compression_range(self, df: pd.DataFrame, num_examples: int) -> pd.DataFrame:
        """Select examples across different compression ratios."""
        df_clean = df.dropna(subset=['compression'])
        if len(df_clean) == 0:
            return df.sample(n=min(num_examples, len(df)), random_state=42)
        
        # Divide compression range into bins
        compression_bins = pd.qcut(df_clean['compression'], q=min(num_examples, 5), duplicates='drop')
        selected_examples = []
        
        for bin_name in compression_bins.cat.categories:
            bin_data = df_clean[compression_bins == bin_name]
            if len(bin_data) > 0:
                # Select one example from each bin (closest to bin median)
                bin_median = bin_data['compression'].median()
                bin_data['distance_to_median'] = abs(bin_data['compression'] - bin_median)
                selected_examples.append(bin_data.nsmallest(1, 'distance_to_median'))
        
        result = pd.concat(selected_examples).head(num_examples)
        return result
    
    def _select_length_range(self, df: pd.DataFrame, num_examples: int) -> pd.DataFrame:
        """Select examples across different original text lengths."""
        # Divide length range into bins
        length_bins = pd.qcut(df['original_length'], q=min(num_examples, 5), duplicates='drop')
        selected_examples = []
        
        for bin_name in length_bins.cat.categories:
            bin_data = df[length_bins == bin_name]
            if len(bin_data) > 0:
                # Select one example from each bin
                selected_examples.append(bin_data.sample(n=1, random_state=42))
        
        result = pd.concat(selected_examples).head(num_examples)
        return result
    
    def format_examples(self, output_format: str = "markdown", 
                       max_original_length: int = None,
                       max_summary_length: int = 300,
                       include_metrics: bool = True,
                       full_original_text: bool = False) -> str:
        """
        Format selected examples for report inclusion.
        
        Args:
            output_format: Output format ('markdown', 'latex', 'plain')
            max_original_length: Maximum original text length to display (None = no limit)
            max_summary_length: Maximum summary length to display
            include_metrics: Whether to include performance metrics
            full_original_text: Whether to show full original text (overrides max_original_length)
        
        Returns:
            Formatted string
        """
        if not self.selected_examples:
            return "No examples selected."
        
        if output_format.lower() == "latex":
            return self._format_latex(max_original_length, max_summary_length, include_metrics, full_original_text)
        elif output_format.lower() == "markdown":
            return self._format_markdown(max_original_length, max_summary_length, include_metrics, full_original_text)
        else:
            return self._format_plain(max_original_length, max_summary_length, include_metrics, full_original_text)
    
    def _clean_text_for_display(self, text: str) -> str:
        """Clean text for better display formatting."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Ensure sentence spacing
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        return text
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to specified length with ellipsis."""
        if not text or max_length is None or len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        if max_length > 10:
            truncated = text[:max_length-3]
            # Find last complete word
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # Only use word boundary if it's not too short
                truncated = truncated[:last_space]
            return truncated + "..."
        else:
            return text[:max_length-3] + "..."
    
    def _format_markdown(self, max_original_length: int, max_summary_length: int, 
                        include_metrics: bool, full_original_text: bool) -> str:
        """Format examples as Markdown."""
        output = ["# Selected Text Summarization Examples\n"]
        
        for i, example in enumerate(self.selected_examples, 1):
            output.append(f"## Example {i}: {example['file_id']}\n")
            
            if include_metrics:
                metrics = []
                metrics.append(f"**Length:** {example['original_length']} → {example['summary_length']} tokens")
                metrics.append(f"**Compression:** {example['compression']:.3f}")
                
                if 'rouge_l' in example:
                    metrics.append(f"**ROUGE-L:** {example['rouge_l']:.3f}")
                if 'bertscore_recall' in example:
                    metrics.append(f"**BERTScore:** {example['bertscore_recall']:.3f}")
                
                # Show actual word counts if available
                if 'actual_original_words' in example:
                    metrics.append(f"**Actual words:** {example['actual_original_words']} → {example['actual_summary_words']}")
                
                output.append(" | ".join(metrics) + "\n")
            
            # Handle original text
            if 'original_text' in example and example['original_text']:
                orig_text = example['original_text']
                if not full_original_text and max_original_length is not None:
                    orig_text = self._truncate_text(orig_text, max_original_length)
                # Clean up text for better readability
                orig_text = self._clean_text_for_display(orig_text)
                output.append(f"### Original Text\n```\n{orig_text}\n```\n")
            else:
                output.append("### Original Text\n*Text content not available*\n")
            
            # Handle summary text
            if 'summary_text' in example and example['summary_text']:
                summ_text = self._truncate_text(example['summary_text'], max_summary_length)
                summ_text = self._clean_text_for_display(summ_text)
                output.append(f"### Generated Summary\n```\n{summ_text}\n```\n")
            else:
                output.append("### Generated Summary\n*Summary content not available*\n")
            
            output.append("---\n")
        
        return "\n".join(output)
    
    def _format_latex(self, max_original_length: int, max_summary_length: int, 
                     include_metrics: bool, full_original_text: bool) -> str:
        """Format examples as LaTeX."""
        output = ["\\section{Selected Text Summarization Examples}\n"]
        
        for i, example in enumerate(self.selected_examples, 1):
            output.append(f"\\subsection{{Example {i}: \\texttt{{{example['file_id']}}}}}\n")
            
            if include_metrics:
                output.append("\\begin{itemize}")
                output.append(f"\\item Length: {example['original_length']} $\\rightarrow$ {example['summary_length']} tokens")
                output.append(f"\\item Compression Ratio: {example['compression']:.3f}")
                
                if 'rouge_l' in example:
                    output.append(f"\\item ROUGE-L Score: {example['rouge_l']:.3f}")
                if 'bertscore_recall' in example:
                    output.append(f"\\item BERTScore Recall: {example['bertscore_recall']:.3f}")
                
                if 'actual_original_words' in example:
                    output.append(f"\\item Actual Word Count: {example['actual_original_words']} $\\rightarrow$ {example['actual_summary_words']} words")
                
                output.append("\\end{itemize}\n")
            
            # Handle original text
            if 'original_text' in example and example['original_text']:
                orig_text = example['original_text']
                if not full_original_text and max_original_length is not None:
                    orig_text = self._truncate_text(orig_text, max_original_length)
                orig_text = self._clean_text_for_display(orig_text)
                # Escape LaTeX special characters
                orig_text = self._escape_latex(orig_text)
                output.append("\\paragraph{Original Text:}")
                output.append("\\begin{quote}")
                output.append("\\small")
                output.append(orig_text)
                output.append("\\end{quote}\n")
            else:
                output.append("\\paragraph{Original Text:} \\textit{Text content not available}\n")
            
            # Handle summary text
            if 'summary_text' in example and example['summary_text']:
                summ_text = self._truncate_text(example['summary_text'], max_summary_length)
                summ_text = self._clean_text_for_display(summ_text)
                summ_text = self._escape_latex(summ_text)
                output.append("\\paragraph{Generated Summary:}")
                output.append("\\begin{quote}")
                output.append("\\small")
                output.append(summ_text)
                output.append("\\end{quote}\n")
            else:
                output.append("\\paragraph{Generated Summary:} \\textit{Summary content not available}\n")
            
            # Add some spacing between examples
            if i < len(self.selected_examples):
                output.append("\\vspace{0.5cm}\n")
        
        return "\n".join(output)
    
    def _format_plain(self, max_original_length: int, max_summary_length: int, 
                     include_metrics: bool, full_original_text: bool) -> str:
        """Format examples as plain text."""
        output = ["SELECTED TEXT SUMMARIZATION EXAMPLES", "=" * 50, ""]
        
        for i, example in enumerate(self.selected_examples, 1):
            output.append(f"EXAMPLE {i}: {example['file_id']}")
            output.append("-" * 40)
            
            if include_metrics:
                output.append(f"Length: {example['original_length']} → {example['summary_length']} tokens")
                output.append(f"Compression Ratio: {example['compression']:.3f}")
                
                if 'rouge_l' in example:
                    output.append(f"ROUGE-L Score: {example['rouge_l']:.3f}")
                if 'bertscore_recall' in example:
                    output.append(f"BERTScore Recall: {example['bertscore_recall']:.3f}")
                
                if 'actual_original_words' in example:
                    output.append(f"Actual Words: {example['actual_original_words']} → {example['actual_summary_words']} words")
                output.append("")
            
            # Handle original text
            if 'original_text' in example and example['original_text']:
                orig_text = example['original_text']
                if not full_original_text and max_original_length is not None:
                    orig_text = self._truncate_text(orig_text, max_original_length)
                orig_text = self._clean_text_for_display(orig_text)
                wrapped_text = textwrap.fill(orig_text, width=80, initial_indent="  ", subsequent_indent="  ")
                output.append("ORIGINAL TEXT:")
                output.append(wrapped_text)
                output.append("")
            else:
                output.append("ORIGINAL TEXT: [Not available]")
                output.append("")
            
            # Handle summary text
            if 'summary_text' in example and example['summary_text']:
                summ_text = self._truncate_text(example['summary_text'], max_summary_length)
                summ_text = self._clean_text_for_display(summ_text)
                wrapped_text = textwrap.fill(summ_text, width=80, initial_indent="  ", subsequent_indent="  ")
                output.append("GENERATED SUMMARY:")
                output.append(wrapped_text)
                output.append("")
            else:
                output.append("GENERATED SUMMARY: [Not available]")
                output.append("")
            
            output.append("=" * 50)
            output.append("")
        
        return "\n".join(output)
    
    def _escape_latex(self, text: str) -> str:
        """Escape LaTeX special characters."""
        chars_to_escape = {'&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', 
                          '^': '\\textasciicircum{}', '_': '\\_', '{': '\\{', '}': '\\}',
                          '~': '\\textasciitilde{}', '\\': '\\textbackslash{}'}
        
        for char, escaped in chars_to_escape.items():
            text = text.replace(char, escaped)
        return text
    
    def print_selection_summary(self):
        """Print a summary of the selected examples."""
        if not self.selected_examples:
            print("No examples selected.")
            return
        
        print(f"\nSELECTED EXAMPLES SUMMARY")
        print("=" * 40)
        print(f"Number of examples: {len(self.selected_examples)}")
        
        # Check text availability
        examples_with_text = sum(1 for ex in self.selected_examples 
                               if ex.get('original_text', '').strip() and ex.get('summary_text', '').strip())
        print(f"Examples with full text content: {examples_with_text}/{len(self.selected_examples)}")
        
        # Calculate statistics
        lengths = [ex['original_length'] for ex in self.selected_examples]
        compressions = [ex['compression'] for ex in self.selected_examples]
        
        print(f"Original length range: {min(lengths)} - {max(lengths)} tokens")
        print(f"Average compression: {np.mean(compressions):.3f}")
        print(f"Compression range: {min(compressions):.3f} - {max(compressions):.3f}")
        
        # Show actual word counts if available
        if any('actual_original_words' in ex for ex in self.selected_examples):
            actual_orig = [ex['actual_original_words'] for ex in self.selected_examples 
                          if 'actual_original_words' in ex]
            actual_summ = [ex['actual_summary_words'] for ex in self.selected_examples 
                          if 'actual_summary_words' in ex]
            if actual_orig and actual_summ:
                print(f"Actual word counts - Original: {min(actual_orig)}-{max(actual_orig)}, "
                     f"Summary: {min(actual_summ)}-{max(actual_summ)}")
        
        if any('rouge_l' in ex for ex in self.selected_examples):
            rouge_scores = [ex['rouge_l'] for ex in self.selected_examples if 'rouge_l' in ex]
            print(f"ROUGE-L range: {min(rouge_scores):.3f} - {max(rouge_scores):.3f}")
        
        if any('bertscore_recall' in ex for ex in self.selected_examples):
            bert_scores = [ex['bertscore_recall'] for ex in self.selected_examples if 'bertscore_recall' in ex]
            print(f"BERTScore range: {min(bert_scores):.3f} - {max(bert_scores):.3f}")
        
        print(f"\nSelected IDs: {', '.join([ex['file_id'] for ex in self.selected_examples])}")
        
        # Show a preview of text lengths
        if examples_with_text > 0:
            print(f"\nText content preview:")
            for ex in self.selected_examples[:3]:  # Show first 3 examples
                if ex.get('original_text', '').strip():
                    orig_len = len(ex['original_text'])
                    summ_len = len(ex['summary_text'])
                    orig_preview = ex['original_text'][:100] + "..." if orig_len > 100 else ex['original_text']
                    summ_preview = ex['summary_text'][:50] + "..." if summ_len > 50 else ex['summary_text']
                    print(f"  {ex['file_id']}: Original={orig_len} chars, Summary={summ_len} chars")
                    print(f"    '{orig_preview}' → '{summ_preview}'")

def main():
    parser = argparse.ArgumentParser(
        description='Select representative text-summary pairs for academic reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Selection Strategies:
  quality_range     - Select best, worst, and middle quality examples
  compression_range - Select examples across different compression ratios
  length_range      - Select examples across different text lengths
  random           - Random selection
  best             - Select best performing examples
  worst            - Select worst performing examples

Output Formats:
  markdown         - Markdown format for documentation
  latex           - LaTeX format for academic papers
  plain           - Plain text format
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data_file', '-d',
        type=str,
        required=True,
        help='Path to analysis data CSV file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--original_dir',
        type=str,
        help='Directory containing original texts (required if text not in CSV)'
    )
    
    parser.add_argument(
        '--summary_dir',
        type=str,
        help='Directory containing summaries (required if text not in CSV)'
    )
    
    parser.add_argument(
        '--num_examples', '-n',
        type=int,
        default=6,
        help='Number of examples to select (default: 6)'
    )
    
    parser.add_argument(
        '--selection_strategy', '-s',
        type=str,
        default='quality_range',
        choices=['quality_range', 'compression_range', 'length_range', 'random', 'best', 'worst'],
        help='Selection strategy (default: quality_range)'
    )
    
    parser.add_argument(
        '--specific_ids',
        type=str,
        help='Comma-separated list of specific file IDs to select'
    )
    
    parser.add_argument(
        '--output_format', '-f',
        type=str,
        default='markdown',
        choices=['markdown', 'latex', 'plain'],
        help='Output format (default: markdown)'
    )
    
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        help='Output file (default: print to stdout)'
    )
    
    parser.add_argument(
        '--max_original_length',
        type=int,
        help='Maximum original text length to display (default: no limit if --full_original_text)'
    )
    
    parser.add_argument(
        '--max_summary_length',
        type=int,
        default=300,
        help='Maximum summary length to display (default: 300)'
    )
    
    parser.add_argument(
        '--full_original_text',
        action='store_true',
        help='Show full original text without truncation (overrides --max_original_length)'
    )
    
    parser.add_argument(
        '--min_length',
        type=int,
        default=50,
        help='Minimum original text length for selection (default: 50)'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=1000,
        help='Maximum original text length for selection (default: 1000)'
    )
    
    parser.add_argument(
        '--score_column',
        type=str,
        default='rouge_l',
        help='Score column to use for quality-based selection (default: rouge_l)'
    )
    
    parser.add_argument(
        '--no_metrics',
        action='store_true',
        help='Exclude performance metrics from output'
    )
    
    args = parser.parse_args()
    
    # Parse specific IDs
    specific_ids = None
    if args.specific_ids:
        specific_ids = [id.strip() for id in args.specific_ids.split(',')]
    
    try:
        # Initialize selector
        selector = ExampleSelector(
            data_file=args.data_file,
            original_dir=args.original_dir,
            summary_dir=args.summary_dir
        )
        
        # Load data
        selector.load_data()
        
        # Check if we have text content
        has_text = hasattr(selector, 'df') and selector.df is not None
        if has_text:
            text_available = (
                'original_text' in selector.df.columns and 
                'summary_text' in selector.df.columns and
                not selector.df['original_text'].isna().all() and
                not selector.df['summary_text'].isna().all()
            )
            
            if not text_available:
                print("\nWarning: No text content found!")
                print("The selected examples will only include metrics, not actual text content.")
                print("To include full text content:")
                print("1. Ensure your CSV has 'original_text' and 'summary_text' columns, OR")
                print("2. Provide --original_dir and --summary_dir arguments")
                response = input("\nContinue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("Exiting. Please provide text content sources.")
                    sys.exit(0)
        
        # Select examples
        examples = selector.select_examples(
            num_examples=args.num_examples,
            strategy=args.selection_strategy,
            specific_ids=specific_ids,
            min_length=args.min_length,
            max_length=args.max_length,
            score_column=args.score_column
        )
        
        # Print selection summary
        selector.print_selection_summary()
        
        # Determine max_original_length based on args
        max_original_length = None if args.full_original_text else args.max_original_length
        
        # Show notification if using full text
        if args.full_original_text:
            print(f"\n✓ Using full original text (no truncation)")
        elif max_original_length:
            print(f"\n✓ Truncating original text at {max_original_length} characters")
        else:
            print(f"\n✓ Using full original text (no length limit specified)")
            
        if args.max_summary_length:
            print(f"✓ Truncating summaries at {args.max_summary_length} characters")
        
        # Format output
        formatted_output = selector.format_examples(
            output_format=args.output_format,
            max_original_length=max_original_length,
            max_summary_length=args.max_summary_length,
            include_metrics=not args.no_metrics,
            full_original_text=args.full_original_text
        )
        
        # Save or print output
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            print(f"\nExamples saved to: {output_path}")
        else:
            print("\n" + "="*50)
            print("FORMATTED OUTPUT:")
            print("="*50)
            print(formatted_output)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()