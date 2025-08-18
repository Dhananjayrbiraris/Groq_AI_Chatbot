# nlu.py
from groq import Groq
from dotenv import load_dotenv
import os
import json
from normalize import expand_seniority

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment")

client = Groq(api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """
You are a query-to-filter parser for a people/company dataset.
Return STRICT JSON with keys only from this schema when present in the user text:
{
  "Department": <string>,
  "Seniority": <string>,
  "Job Title": <string>,
  "Location": {
    "Country": <string|null>,
    "State": <string|null>,
    "City": <string|null>
  },
  "Industry": <string|null>,
  "Company Name": <string|null>,
  "Extras": <array of strings>
}
Rules:
- If a field is absent, omit it (do NOT add null unless inside Location).
- Deduce obvious synonyms (e.g., "IT" -> "Information Technology"; "US" -> "United States").
- Seniority should be extracted exactly as spoken. 
    Examples: 
      - "decision makers" → "decision makers"
      - "executives" → "executives"
      - "leadership" → "leadership"
- Do NOT force seniority into a single fixed label. Keep the user wording.
- Later stages will expand these terms into multiple seniority levels.
- If user says "team" after a department, just map to the department (ignore "team").
- Never include commentary. Output ONLY a minified JSON object.
"""


def normalize_filters(filters: dict) -> dict:
    """Normalize raw filters using expansion rules."""
    if "Seniority" in filters:
        filters["Seniority"] = expand_seniority(filters["Seniority"])
    return filters


def extract_filters(query: str) -> dict:
    """Call Groq LLM to convert a query into normalized filter JSON."""
    resp = client.chat.completions.create(
        model="llama3-70b-8192",  # or llama3-8b-8192
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])
