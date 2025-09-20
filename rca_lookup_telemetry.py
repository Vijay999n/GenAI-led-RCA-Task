import argparse
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from datetime import datetime

client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

############################################
# Loaders
############################################
def load_data(incidents_path, kedb_path, telemetry_path):
    """
    Load incidents, KEDB, and telemetry logs.
    """
    inc = pd.read_csv(incidents_path, dtype=str, keep_default_na=False)
    kedb = pd.read_csv(kedb_path, dtype=str, keep_default_na=False)

    with open(telemetry_path, "r") as f:
        telemetry = json.load(f)

    return inc, kedb, telemetry


############################################
# KEDB Embedding Builder
############################################
def build_kedb_embeddings(kedb, model):
    knowledge_base = []

    for _, row in kedb.iterrows():
        sig = row["signature"]
        phrases = [sig]

        if "example_phrases" in row and row["example_phrases"]:
            extra = [p.strip() for p in row["example_phrases"].split(";") if p.strip()]
            phrases.extend(extra)

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


############################################
# RCA Finder (KEDB)
############################################
def find_rca(description, knowledge_base, model, threshold=0.5):
    query_embedding = model.encode(description, convert_to_tensor=True)
    best_score, best_entry = -1, None

    for entry in knowledge_base:
        cos_scores = util.cos_sim(query_embedding, entry["embeddings"])[0]
        score = float(cos_scores.max())

        if score > best_score:
            best_score = score
            best_entry = entry

    return (best_entry, best_score) if best_score >= threshold else (None, best_score)


############################################
# Telemetry Search
############################################
def search_telemetry(description, telemetry_data, model, threshold=0.5):
    """
    Search telemetry logs for possible matches with incident description.
    """
    query_embedding = model.encode(description, convert_to_tensor=True)
    best_score, best_entry = -1, None

    for trace in telemetry_data:
        for span in trace["spans"]:
            span_text = span.get("message", "") + " " + span.get("operation", "")
            span_embedding = model.encode(span_text, convert_to_tensor=True)
            score = float(util.cos_sim(query_embedding, span_embedding)[0][0])

            if score > best_score:
                best_score = score
                best_entry = {
                    "trace_id": trace["trace_id"],
                    "service": span.get("service"),
                    "component": span.get("component"),
                    "status": span.get("status"),
                    "severity": span.get("tags", {}).get("severity", "N/A"),
                    "start_time": span.get("start_time"),
                    "message": span.get("message"),
                    "tags": span.get("tags"),
                }

    return (best_entry, best_score) if best_score >= threshold else (None, best_score)


############################################
# LLM Suggestion
############################################
def get_llm_suggestion(description, telemetry_context=None):
    try:
        context_str = ""
        if telemetry_context:
            context_str = (
                f"\nTelemetry Evidence:\n"
                f"- Trace ID: {telemetry_context['trace_id']}\n"
                f"- Service: {telemetry_context['service']}\n"
                f"- Component: {telemetry_context['component']}\n"
                f"- Status: {telemetry_context['status']} ({telemetry_context['severity']})\n"
                f"- Time: {telemetry_context['start_time']}\n"
                f"- Message: {telemetry_context['message']}\n"
                f"- Tags: {telemetry_context['tags']}\n"
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
                {
                    "role": "user",
                    "content": f"Incident: {description}{context_str}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error calling LLM: {e}"


############################################
# Main
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="Incident description")
    parser.add_argument("--incidents_csv", default="incidents.csv", help="Path to incidents CSV")
    parser.add_argument("--kedb_csv", default="kedb.csv", help="Path to KEDB CSV")
    parser.add_argument("--telemetry_json", default="telemetry_log.json", help="Path to telemetry logs JSON")
    args = parser.parse_args()

    # Load data
    incidents, kedb, telemetry = load_data(args.incidents_csv, args.kedb_csv, args.telemetry_json)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 1: Check KEDB
    print("searching the possible match in KEDB...")
    knowledge_base = build_kedb_embeddings(kedb, model)
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
        # Step 2: Check Telemetry
        print("\n🔍 RCA Not Identified in KEDB, searching telemetry logs...")
        telemetry_entry, t_score = search_telemetry(args.description, telemetry, model)

        if telemetry_entry:
            print("\n⚡ Incident correlated with telemetry data")
            print("input_description:", args.description)
            print("similarity:", round(t_score, 3))
            print("affected_service:", telemetry_entry["service"])
            print("component:", telemetry_entry["component"])
            print("status:", telemetry_entry["status"])
            print("severity:", telemetry_entry["severity"])
            print("timestamp:", telemetry_entry["start_time"])
            print("message:", telemetry_entry["message"])
            print("tags:", telemetry_entry["tags"])
            print("🧠 Calling LLM with telemetry context...")

            suggestion = get_llm_suggestion(args.description, telemetry_entry)
            print("🔍 Below is the LLM Suggestion:\n", suggestion.replace("*", ""))

        else:
            # Step 3: Pure LLM
            print("\n❓ RCA Not Found in Telemetry either")
            print("🧠 Calling LLM for generic suggestion...")
            suggestion = get_llm_suggestion(args.description)
            print("🔍 Below is the LLM Suggestion:\n", suggestion.replace("*", ""))
