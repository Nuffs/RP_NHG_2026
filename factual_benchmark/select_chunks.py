import json
import random
import spacy

nlp = spacy.load("nl_core_news_sm")

def chunck_text_on_sentences(text, max_tokens=400, overlap_sentences=1):
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

def sample_chunks_per_guideline(all_chunks, n=3, seed=42):
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