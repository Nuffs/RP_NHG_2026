import json
import random
import spacy

nlp = spacy.load("nl_core_news_sm")

# Load all chuncks from the scraped data
with open("data/nhg_subset_guidelines.jsonl", "r") as f:
    all_raw_chunks = json.load(f)


# def priority_score(chunk):
#     """ 
#     Rule-based selection for round-trip evaluation.
#     Criteria based on literature (Demner-Fushman et al.)
#     - Relevantie: medische termen, aanbevelingen, diagnostiek, beleid
#     - Informatiedichtheid: aantal tokens (maar niet te lang)
#     - Diversiteit: verschillende secties van de richtlijn
#     """
#     priority_keywords = ["aanbeveling", "diagnostiek", "beleid", "medicatie", "behandeling",
#                         "indicatie", "contra-indicatie", "medicament", "evaluatie", "dosering", "streefwaarde"]
    
#     text = chunk["text"].lower()


