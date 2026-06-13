"""
select_chunks
-------------

Utilities for chunking raw guidelines into sentence-level chunks and
sampling subsets per guideline. The module exposes helpers used during
data preparation: `chunck_text_on_sentences`, `chunk_document`, and
sampling functions.
"""

import json
import random
import spacy
import argparse
import os
from pathlib import Path

nlp = spacy.load("nl_core_news_sm")

def chunck_text_on_sentences(text, max_tokens=400, overlap_sentences=1):
    """
    Chunk text into sentence-based segments with optional overlap.
    
    Args:
        text (str): The input text.
        max_tokens (int, optional): Maximum number of tokens (words) per chunk. Defaults to 400.
        overlap_sentences (int, optional): Number of sentences to overlap between chunks. Defaults to 1.
    
    Returns:
        list: List of text chunks (strings), each approximately max_tokens words long.
    
    Example:
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> chunks = chunck_text_on_sentences(text, max_tokens=10, overlap_sentences=1)
        >>> len(chunks)  # Number of chunks created
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    curr_length = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())

        if curr_length + sentence_tokens > max_tokens:
            # save curr chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            # start new chunk with overlap
            current_chunk = current_chunk[-overlap_sentences:] + [sentence]
            curr_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            curr_length += sentence_tokens
    
    # add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    
    return chunks

def chunk_document(raw_chunk, max_tokens=400):
    """
    Rechunk a raw document chunk using sentence-based segmentation.
    
    Args:
        raw_chunk (dict): Input chunk dictionary with keys:
            - 'text' (str): The text content to rechunk
            - 'doc_id' (str): Document identifier
            - 'chunk_id' (str): Original chunk identifier
            - 'section_path' (list): Hierarchy path of sections
        max_tokens (int, optional): Maximum tokens per rechunked segment. Defaults to 400.
    
    Returns:
        list: List of rechunked item dictionaries with keys:
            - 'doc_id' (str): Same as input
            - 'chunk_id' (str): Updated with _XXXX suffix indicating position
            - 'section_path' (list): Same as input
            - 'text' (str): Rechunked text segment
            - 'tokens' (int): Word count of this segment
    """
    text = raw_chunk["text"]
    sentences = chunck_text_on_sentences(text, max_tokens=max_tokens)
    
    chunks = []
    for i, chunk in enumerate(sentences, start=1):
        chunks.append({
            "doc_id": raw_chunk["doc_id"],
            "chunk_id": f"{raw_chunk['chunk_id']}_{i:04d}",
            "section_path": raw_chunk["section_path"],
            "text": chunk,
            "tokens": len(chunk.split())
        })
    return chunks

def sample_chunks_per_guideline(all_chunks, n=20):
    """
    Sample the n longest chunks (by token count) from each guideline.
    """
    chunks_by_doc = {}
    for chunk in all_chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)

    sampled_chunks = []
    for doc_id, chunks in chunks_by_doc.items():
        sorted_chunks = sorted(chunks, key=lambda x: x["tokens"], reverse=True)
        sampled_chunks.extend(sorted_chunks[:n])

    return sampled_chunks

def sample_chunks_per_guideline_prompt_experiment(all_chunks, n=3, seed=42):
    random.seed(seed)
    chunks_by_doc = {}
    
    for chunk in all_chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)
    
    sampled_chunks = []
    for doc_id, chunks in chunks_by_doc.items():
        sampled_chunks.extend(random.sample(chunks, min(n, len(chunks))))
    
    return sampled_chunks


if __name__ == "__main__":
    with open("data/nhg_subset_guidelines.jsonl", "r") as f:
        raw_chunks = [json.loads(line) for line in f]

    all_chunks = []
    for raw_chunk in raw_chunks:
        chunks = chunk_document(raw_chunk)
        all_chunks.extend(chunks)
    
    sampled = sample_chunks_per_guideline(all_chunks)
    
    with open("data/benchmark_chunks.jsonl", "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved benchmark chunks to {"data/benchmark_chunks.jsonl"}")