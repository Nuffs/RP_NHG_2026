import json

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "context_blocks"  # Vul je echte collectienaam in

# Haal alle 2500 records op (we zetten de limiet lekker hoog op 3000)
records, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=3000,
    with_payload=True,
    with_vectors=False
)

# Trek alleen de data (payload) eruit en zet het in een lijst
all_chunks = [record.payload for record in records]

# Opslaan als JSON-bestand
with open("alle_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4, ensure_ascii=False)

print(f"Succes! {len(all_chunks)} chunks opgeslagen in 'alle_chunks.json'.")