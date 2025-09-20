import argparse
import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# -------------------- CONFIG --------------------
client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

KEDB_INDEX_FILE = "kedb.index"
KEDB_META_FILE = "kedb_meta.json"
TELEMETRY_INDEX_FILE = "telemetry.index"
TELEMETRY_META_FILE = "telemetry_meta.json"

# -------------------- UTILS --------------------
def save_faiss_index(index, index_file, metadata, meta_file):
    faiss.write_index(index, index_file)
    with open(meta_file, "w") as f:
        json.dump(metadata, f)


def load_faiss_index(index_file, meta_file):
    index = faiss.read_index(index_file)
    with open(meta_file, "r") as f:
        metadata = json.load(f)
    return index, metadata


# -------------------- DATA LOADING --------------------
def load_data(incidents_path, kedb_path, telemetry_path):
    inc = pd.read_csv(incidents_path, dtype=str, keep_default_na=False)
    kedb = pd.read_csv(kedb_path, dtype=str, keep_default_na=False)
    with open(telemetry_path, "r") as f:
        telemetry = json.load(f)
    return inc, kedb, telemetry


# -------------------- FAISS HELPERS --------------------
def build_faiss_index(texts, model):
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


# -------------------- KEDB --------------------
def prepare_kedb(kedb, model):
    if os.path.exists(KEDB_INDEX_FILE) and os.path.exists(KEDB_META_FILE):
        return load_faiss_index(KEDB_INDEX_FILE, KEDB_META_FILE)

    texts, metadata = [], {}
    idx = 0

    for _, row in kedb.iterrows():
        sig = row["signature"]
        phrases = [sig]
        if row.get("example_phrases"):
            phrases.extend([p.strip() for p in row["example_phrases"].split(";") if p.strip()])

        for phrase in phrases:
            texts.append(phrase)
            metadata[str(idx)] = {
                "signature": sig,
                "rca": row["rca"],
                "workaround": row["workaround"],
                "permanent_fix": row["permanent_fix"],
                "last_reviewed": row["last_reviewed"]
            }
            idx += 1

    index, _ = build_faiss_index(texts, model)
    save_faiss_index(index, KEDB_INDEX_FILE, metadata, KEDB_META_FILE)
    return index, metadata


# -------------------- TELEMETRY --------------------
def prepare_telemetry(telemetry, model):
    if os.path.exists(TELEMETRY_INDEX_FILE) and os.path.exists(TELEMETRY_META_FILE):
        return load_faiss_index(TELEMETRY_INDEX_FILE, TELEMETRY_META_FILE)

    texts, metadata = [], {}
    idx = 0

    for trace in telemetry:
        for span in trace["spans"]:
            text = span["message"] + " " + span["operation"]
            texts.append(text)
            metadata[str(idx)] = {
                "trace_id": trace["trace_id"],
                "span_id": span["span_id"],
                "service": span["service"],
                "component": span["component"],
                "status": span["status"],
                "tags": span["tags"],
                "message": span["message"],
                "start_time": span["start_time"]
            }
            idx += 1

    index, _ = build_faiss_index(texts, model)
    save_faiss_index(index, TELEMETRY_INDEX_FILE, metadata, TELEMETRY_META_FILE)
    return index, metadata


# -------------------- SEARCH --------------------
def search_index(description, index, metadata, model, threshold=0.5):
    q_emb = model.encode([description], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q_emb, k=1)
    score, idx = scores[0][0], ids[0][0]
    if score >= threshold and str(idx) in metadata:
        return metadata[str(idx)], float(score)
    return None, float(score)


# -------------------- LLM --------------------
def get_llm_suggestion(description, telemetry_context=None):
    try:
        context = ""
        if telemetry_context:
            context = (
                f"Telemetry Evidence:\n"
                f"Service: {telemetry_context['service']} | Component: {telemetry_context['component']}\n"
                f"Status: {telemetry_context['status']} | Severity: {telemetry_context['tags'].get('severity', 'N/A')}\n"
                f"Time: {telemetry_context['start_time']}\n"
                f"Message: {telemetry_context['message']}\n"
                f"Tags: {telemetry_context['tags']}\n"
            )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            top_p=0.1,
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert incident analyst. "
                        "Respond in a structured format with these sections: "
                        "Incident Summary, Root Cause, Workaround, Permanent Fix, and Preventative Measures. "
                        "Use bullet points or numbered lists. Be concise and factual."
                    )
                },
                {"role": "user", "content": f"Incident: {description}\n{context}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error calling LLM: {e}"


# -------------------- MAIN --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="Incident description")
    parser.add_argument("--incidents_csv", default="incidents.csv", help="Path to incidents CSV")
    parser.add_argument("--kedb_csv", default="kedb.csv", help="Path to KEDB CSV")
    parser.add_argument("--telemetry_json", default="telemetry_log.json", help="Path to telemetry logs JSON")
    args = parser.parse_args()

    # Load data
    incidents, kedb, telemetry = load_data(args.incidents_csv, args.kedb_csv, args.telemetry_json)

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Prepare indexes
    kedb_index, kedb_meta = prepare_kedb(kedb, model)
    telemetry_index, telemetry_meta = prepare_telemetry(telemetry, model)

    # Step 1: Search KEDB
    rca_entry, score = search_index(args.description, kedb_index, kedb_meta, model)
    print("searching the possible match in KEDB...")
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
        print("\n🔍 RCA Not Identified in KEDB, searching telemetry logs...")
        telemetry_entry, t_score = search_index(args.description, telemetry_index, telemetry_meta, model)
        if telemetry_entry:
            print("\n⚡ Incident correlated with telemetry data")
            print("input_description:", args.description)
            print("similarity:", round(t_score, 3))
            print("affected_service:", telemetry_entry["service"])
            print("component:", telemetry_entry["component"])
            print("status:", telemetry_entry["status"])
            print("severity:", telemetry_entry["tags"].get("severity", "N/A"))
            print("timestamp:", telemetry_entry["start_time"])
            print("message:", telemetry_entry["message"])
            print("tags:", telemetry_entry["tags"])
            print("🧠 Calling LLM with telemetry context...")
            suggestion = get_llm_suggestion(args.description, telemetry_entry)
            print("🔍 Below is the LLM Suggestion:\n", suggestion)
        else:
            print("\n❓ RCA Not Found in Telemetry either")
            print("🧠 Calling LLM for generic suggestion...")
            suggestion = get_llm_suggestion(args.description)
            print("🔍 Below is the LLM Suggestion:\n", suggestion)
