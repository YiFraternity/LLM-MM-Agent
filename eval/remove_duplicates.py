import json
import hashlib
from pathlib import Path

def deduplicate_jsonl(input_file, output_file=None):
    """
    Remove duplicate lines from a JSONL file based on content.

    Args:
        input_file (str): Path to the input JSONL file
        output_file (str, optional): Path to save the deduplicated output.
                                   If None, will save to input_file + '.deduped.jsonl'
    """
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.deduped.jsonl'))

    seen_hashes = set()
    duplicates_removed = 0
    total_lines = 0

    # First pass: count total lines and find duplicates
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            line_hash = hashlib.md5(line.strip().encode('utf-8')).hexdigest()
            seen_hashes.add(line_hash)

    # Second pass: write unique lines
    seen_hashes = set()
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
            if line_hash not in seen_hashes:
                seen_hashes.add(line_hash)
                outfile.write(line + '\n')
            else:
                duplicates_removed += 1

    unique_lines = total_lines - duplicates_removed
    print(f"Processed {total_lines} lines in total")
    print(f"Removed {duplicates_removed} duplicate lines")
    print(f"Kept {unique_lines} unique lines")
    print(f"Deduplicated file saved to: {output_file}")

if __name__ == "__main__":
    input_file = 'eval/output/2010_D/3.Qwen2.5-14B-Instruct_section_classification_output.jsonl'
    output_file = 'eval/output/2010_D/3.Qwen2.5-14B-Instruct_section_classification_output_deduped.jsonl'

    deduplicate_jsonl(input_file, output_file)
