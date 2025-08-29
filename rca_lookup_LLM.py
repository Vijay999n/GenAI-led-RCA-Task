import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

def load_data(incidents_path, kedb_path):
    """
    Load incidents and KEDB into pandas DataFrames.
    """
    inc = pd.read_csv(incidents_path, dtype=str, keep_default_na=False)
    kedb = pd.read_csv(kedb_path, dtype=str, keep_default_na=False)
    return inc, kedb


def build_kedb_embeddings(kedb, model):
    """
    Build embeddings for KEDB entries.
    For each signature, we use both the signature itself and the example_phrases.
    Returns a list of dicts with embeddings and metadata.
    """
    knowledge_base = []

    for _, row in kedb.iterrows():
        sig = row["signature"]
        # Start with the signature itself
        phrases = [sig]

        # Add example phrases if present
        if "example_phrases" in row and row["example_phrases"]:
            extra = [p.strip() for p in row["example_phrases"].split(";") if p.strip()]
            phrases.extend(extra)

        # Encode all phrases for this signature
        embeddings = model.encode(phrases, convert_to_tensor=True)

        knowledge_base.append({
            "signature": sig,
            "embeddings": embeddings,
            "rca": row["rca"],
            "workaround": row["workaround"],
            "permanent_fix": row["permanent_fix"],
            "last_reviewed": row["last_reviewed"],
        })

    return knowledge_base


def find_rca(description, knowledge_base, model, threshold=0.5):
    """
    Given a description, find the closest RCA in the knowledge base using embeddings.
    """
    query_embedding = model.encode(description, convert_to_tensor=True)

    best_score = -1
    best_entry = None

    for entry in knowledge_base:
        # Compare query with all phrase embeddings of this signature
        cos_scores = util.cos_sim(query_embedding, entry["embeddings"])[0]
        score = float(cos_scores.max())

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= threshold:
        return best_entry, best_score
    else:
        return None, best_score

def get_llm_suggestion(description):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Recommended replacement
            messages=[
                {"role": "system", "content": "You are an expert incident analyst..."},
                {"role": "user", "content": f"Incident: {description}\nSuggest RCA, workaround, and fix."}
            ],
            temperature=0.5,
            max_tokens= 1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error calling LLM: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="Incident description")
    parser.add_argument("--incidents_csv", default="incidents.csv", help="Path to incidents CSV")
    parser.add_argument("--kedb_csv", default="kedb.csv", help="Path to KEDB CSV")
    args = parser.parse_args()

    # Load data
    incidents, kedb = load_data(args.incidents_csv, args.kedb_csv)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Prepare KEDB knowledge base
    knowledge_base = build_kedb_embeddings(kedb, model)

    # Find RCA
    rca_entry, score = find_rca(args.description, knowledge_base, model)
    
    if rca_entry:
        
        print("\n🎯 Known Pattern Detected in KEDB")
        print("input_description:", args.description)
        print("matched_signature:", rca_entry["signature"])
        print("similarity:", round(score, 3))
        print("rca:", rca_entry["rca"])
        print("workaround:", rca_entry["workaround"])
        print("permanent_fix:", rca_entry["permanent_fix"])
        print("last_reviewed:", rca_entry["last_reviewed"])
    else:
        
        print("\n❓ RCA Not Identified in KEDB")
        print("input_description:", args.description)
        print("similarity:", round(score, 3))
        print("🧠 Calling LLM for suggestion...")
        try:
            suggestion = get_llm_suggestion(args.description)
            clean_text = suggestion.replace("*", "")  # remove asterisks if any
            print("🔍 Below is the LLM Suggestion:\n", clean_text)
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
