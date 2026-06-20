import json
import sys
from pathlib import Path

def export_all_active_chunks():
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir))
    
    # Import functions directly from query.py to ensure 100% matching logic
    try:
        from query import load_chunk_texts, load_embedding_index
    except ImportError as e:
        print(f"Error importing from query.py: {e}")
        return 1
        
    print("Loading active chunk IDs from embeddings.json...")
    embeddings_file = current_dir / "embeddings.json"
    try:
        embedding_index = load_embedding_index(embeddings_file)
    except Exception as e:
        print(f"Error loading embedding index: {e}")
        return 1
        
    print(f"Found {len(embedding_index)} chunks in embedding index.")
    
    print("Loading original texts from local .jsonl files using query.py logic...")
    chunk_texts = load_chunk_texts()
    print(f"Loaded {len(chunk_texts)} original chunk texts from local files.")
    
    # Pair them up
    consolidated_chunks = []
    missing_texts_count = 0
    
    for item in embedding_index:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
            
        chunk_data = chunk_texts.get(chunk_id)
        if chunk_data:
            # We have the text and metadata!
            consolidated_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": chunk_data.get("doc_id"),
                "doc_title": chunk_data.get("doc_title"),
                "section_path": chunk_data.get("section_path"),
                "text": chunk_data.get("text"),
                "tokens": chunk_data.get("tokens")
            })
        else:
            missing_texts_count += 1
            # Still include placeholder or chunk ID
            consolidated_chunks.append({
                "chunk_id": chunk_id,
                "warning": "Original text not found in local .jsonl files"
            })
            
    output_path = current_dir.parent / "data" / "all_active_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving consolidated chunks to {output_path.name}...")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_chunks, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully consolidated and exported!")
        print(f"File exported to: {output_path}")
        if missing_texts_count > 0:
            print(f"Note: {missing_texts_count} chunks did not have original text files locally available.")
    except Exception as e:
        print(f"Error saving consolidated file: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(export_all_active_chunks())
