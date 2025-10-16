import json
import os
import argparse
from typing import List, Optional, Tuple

def find_matching_files(tex_dir: str, json_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find matching LaTeX and JSON files in the given directories.
    Traverses the LaTeX directory and finds corresponding JSON files.

    Args:
        tex_dir: Directory containing LaTeX files
        json_dir: Directory containing JSON files

    Returns:
        List of tuples (tex_path, json_path, output_path)
    """
    matches = []

    # Walk through the LaTeX directory to find all .tex files
    for root, _, files in os.walk(tex_dir):
        for file in files:
            if file.endswith('.tex') and file.endswith('_cleaned.tex'):
                # Get relative path from tex_dir
                rel_path = os.path.relpath(root, tex_dir)
                tex_path = os.path.join(root, file)

                # Construct corresponding JSON file path
                json_filename = f"{os.path.splitext(file)[0]}.tex.placeholders.json"
                json_path = os.path.join(json_dir, rel_path, json_filename)

                # Create output path in a directory at the same level as tex_dir
                base_dir = os.path.dirname(tex_dir.rstrip(os.sep))
                output_dir = os.path.join(base_dir, f"{os.path.basename(tex_dir.rstrip(os.sep))}_processed")

                # Create the relative path structure
                rel_path = os.path.relpath(root, tex_dir)
                output_dir = os.path.join(output_dir, rel_path)
                os.makedirs(output_dir, exist_ok=True)

                # Keep the same filename
                output_path = os.path.join(output_dir, file)

                matches.append((tex_path, json_path, output_path))

    return matches

def replace_placeholders(tex_file_path: str, json_file_path: str, output_file_path: Optional[str] = None) -> bool:
    """
    Replace placeholders in a LaTeX file with content from a JSON file.

    Args:
        tex_file_path: Path to the LaTeX file with placeholders
        json_file_path: Path to the JSON file with placeholder content
        output_file_path: Path to save the output file.
                         If None, will overwrite the original file.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            placeholders = json.load(f)

        # Read the LaTeX file
        with open(tex_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace placeholders
        for placeholder, replacement in placeholders.items():
            # Create a simple string replacement for the placeholder
            # We use string replace instead of regex to avoid issues with special characters
            placeholder_str = f"[{placeholder}]"
            content = content.replace(placeholder_str, replacement)

        # Determine output path
        if output_file_path is None:
            output_file_path = tex_file_path

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Write the modified content to the output file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ Successfully processed: {os.path.basename(tex_file_path)}")
        return True

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON {json_file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing {tex_file_path}: {e}")
        return False

def process_all_files(tex_dir: str, json_dir: str):
    """
    Process all LaTeX files in the tex_dir, finding matching JSON files in json_dir.
    """
    print(f"🔍 Searching for matching files in:\n  LaTeX: {tex_dir}\n  JSON:  {json_dir}")

    matches = find_matching_files(tex_dir, json_dir)

    if not matches:
        print("❌ No matching LaTeX and JSON files found.")
        return

    print(f"\nFound {len(matches)} file pairs to process:")
    for i, (tex_path, json_path, output_path) in enumerate(matches, 1):
        print(f"{i}. {os.path.relpath(tex_path, tex_dir)} -> {os.path.relpath(json_path, json_dir)}")

    print("\n🚀 Starting processing...")
    success_count = 0

    for tex_path, json_path, output_path in matches:
        if os.path.exists(tex_path):
            if replace_placeholders(tex_path, json_path, output_path):
                success_count += 1
        else:
            print(f"❌ LaTeX file not found: {tex_path}")

    print(f"\n✅ Processing complete!")
    print(f"   Successfully processed: {success_count}/{len(matches)} files")
    if success_count < len(matches):
        print(f"   Failed: {len(matches) - success_count} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Replace placeholders in LaTeX files with content from JSON files.'
    )

    # For batch processing
    parser.add_argument(
        '--tex-dir',
        help='Directory containing LaTeX files (for batch processing)',
    )
    parser.add_argument(
        '--json-dir',
        help='Directory containing JSON files (for batch processing)',
    )

    # For single file processing
    parser.add_argument(
        '--tex-file',
        default='best_paper_tex_clean/2007_C/2_cleaned.tex',
        help='Path to a single LaTeX file to process'
    )
    parser.add_argument(
        '--json-file',
        default='best_paper_tex_clean/2007_C/2_cleaned.tex.placeholders.json',
        help='Path to a single JSON file with placeholders'
    )
    parser.add_argument(
        '-o',
        '--output',
        default='MMBench/CPMCM/BestPaper/2007_C/2.tex',
        help='Output file path (default: add _processed suffix)'
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.tex_file and args.json_file:
        tex_file = os.path.abspath(args.tex_file)
        json_file = os.path.abspath(args.json_file)
        output_file = os.path.abspath(args.output) if args.output else None

        if not os.path.exists(tex_file):
            print(f"❌ LaTeX file not found: {tex_file}")
            exit(1)
        if not os.path.exists(json_file):
            print(f"❌ JSON file not found: {json_file}")
            exit(1)

        replace_placeholders(tex_file, json_file, output_file)
    else:
        tex_dir = os.path.join(script_dir, args.tex_dir)
        json_dir = os.path.join(script_dir, args.json_dir)

        if not os.path.exists(tex_dir):
            print(f"❌ LaTeX directory not found: {tex_dir}")
            exit(1)
        if not os.path.exists(json_dir):
            print(f"❌ JSON directory not found: {json_dir}")
            exit(1)

        process_all_files(tex_dir, json_dir)
