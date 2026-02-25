import argparse
import os
import json
import re


def convert_pdfs_to_qa_chunks(input_folder: str, output_folder: str, sample_count: int = None, chunk_size:int = 4000) -> None:
    """
    Convert all PDF files in input_folder into chunked JSON files saved to output_folder.

    Each JSON file contains a list of dicts with a 'context' key, where each context
    is a merged group of markdown sections (~4000+ chars each).
    Intermediate markdown files are cached alongside the source PDFs.

    Args:
        fast: If True, use pymupdf4llm instead of docling for PDF-to-markdown conversion.
    """
    os.makedirs(output_folder, exist_ok=True)

    fns = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    if not fns:
        print(f"No PDF files found in {input_folder}")
        return

    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()

    for fn in fns:
        pdf_path = os.path.join(input_folder, fn)
        base_name = os.path.splitext(fn)[0]
        md_suffix = '.md'
        md_path = os.path.join(input_folder, base_name + md_suffix)
        out_path = os.path.join(output_folder, base_name + '.json')

        # Step 1: Convert PDF to markdown if not already cached in input folder
        if not os.path.exists(md_path):
            print(f"Converting {fn} to markdown (docling)...")
            result = converter.convert(pdf_path)
            txt = result.document.export_to_markdown()
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(txt)
            print(f"  Saved markdown to {md_path}")
        else:
            print(f"Using cached markdown for {fn}")

        # Step 2: Parse markdown into sections
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        lines = [line for line in lines if not line.strip().startswith('![Image]')]
        lines = [line for line in lines if not line.strip().startswith('![]')]

        sections = []
        current_heading = None
        current_section = []

        for line in lines:
            stripped = line.strip()
            if re.match(r'^#+\s', stripped):
                if current_heading and current_section:
                    sections.append({
                        'heading': current_heading,
                        'content': '\n'.join(current_section)
                    })
                current_heading = stripped
                current_section = []
            else:
                if stripped and current_heading:
                    current_section.append(line)

        if current_heading and current_section:
            sections.append({
                'heading': current_heading,
                'content': '\n'.join(current_section)
            })

        # Step 3: Merge sections until each chunk is at least chunk_size chars
        combined_sections = []
        i = 0
        while i < len(sections):
            current = {
                'heading': sections[i]['heading'],
                'content': sections[i]['content']
            }
            j = i + 1
            while j < len(sections) and len(current['content']) < chunk_size:
                current['content'] += '\n\n' + sections[j]['heading'] + '\n' + sections[j]['content']
                j += 1
            combined_sections.append(current)
            i = j

        print(f"  {fn}: {len(sections)} sections -> {len(combined_sections)} chunks")

        # Step 4: Format and save
        contexts = [{'context': s['heading'] + '\n' + s['content']} for s in combined_sections]
        if sample_count is not None:
            import random
            random.shuffle(contexts)
            contexts = contexts[:sample_count]

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, indent=2)

        print(f"  Saved {len(contexts)} chunks to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PDF files into chunked JSON context files.')
    parser.add_argument('input_folder', help='Folder containing PDF files to convert')
    parser.add_argument('output_folder', help='Folder to save the output JSON files')
    parser.add_argument('--fast', action='store_true', help='Use pymupdf4llm instead of docling (faster, lower quality)')
    args = parser.parse_args()
    convert_pdfs_to_qa_chunks(args.input_folder, args.output_folder, fast=args.fast)
