# app.py
import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from nlu import extract_filters
from normalize import (
    normalize_department, normalize_seniority,
    normalize_country, build_location_string
)
from search import Retriever

load_dotenv()

st.set_page_config(page_title="Prompt Filter Chatbot", page_icon="🧭", layout="wide")
st.title("Predictiv Data")

# --- Data loading ---
DATA_PATH = os.path.join("data", "sample_dataset.csv")
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DATA_PATH)
    st.session_state.retriever = Retriever(st.session_state.df).build()

retriever = st.session_state.retriever
df = st.session_state.df

# --- Chat UI ---
if "history" not in st.session_state:
    st.session_state.history = []

def render_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

for msg in st.session_state.history:
    render_message(msg["role"], msg["content"])

user_query = st.chat_input("Ask: e.g., 'IT Directors from United States in healthcare'")
if user_query:
    st.session_state.history.append({"role":"user","content":user_query})
    render_message("user", user_query)

    # NLU extraction
    try:
        filt = extract_filters(user_query)
    except Exception as e:
        filt = {}
        st.warning(f"NLU parse fallback: {e}")

    # Normalization
    Department = normalize_department(filt.get("Department"))
    Seniority  = normalize_seniority(filt.get("Seniority"))
    JobTitle   = filt.get("Job Title")
    loc = filt.get("Location", {}) if isinstance(filt.get("Location"), dict) else {}
    Country = normalize_country(loc.get("Country"))
    State   = loc.get("State")
    City    = loc.get("City")
    LocationLike = build_location_string(Country, State, City)
    Industry = filt.get("Industry")
    Company  = filt.get("Company Name")

    with st.expander("Parsed Filters (JSON)", expanded=False):
        st.code(json.dumps({
            "Department": Department,
            "Seniority": Seniority,
            "Job Title": JobTitle,
            "LocationLike": LocationLike,
            "Industry": Industry,
            "Company Name": Company
        }, ensure_ascii=False), language="json")

    # Filter first
    fdf = retriever.filter_df(
        Department=Department,
        Seniority=Seniority,
        JobTitle=JobTitle,
        LocationLike=LocationLike,
        Industry=Industry,
        Company=Company
    )

    # Rerank semantically on the filtered slice
    results = retriever.semantic_rerank(fdf, user_query, top_k=30)

    # Compose assistant reply
    if results.empty:
        answer = f"**No results** for your query."
        render_message("assistant", answer)
    else:
        # Show brief summary
        n = len(fdf)
        answer = (
            f"**Found {n} candidate rows** after applying filters. "
            f"Top {min(30, len(results))} shown below (semantic order)."
        )
        render_message("assistant", answer)
        st.dataframe(results[
            ["First Name","Last Name","Job Title","Seniority","Parent Department","Company Name",
             "Industry","Location","Office Email ID","LinkedIn URL","similarity_score"]
        ], use_container_width=True)

        # CSV export
        csv = results.to_csv(index=False)
        st.download_button("Download results as CSV", data=csv, file_name="results.csv", mime="text/csv")
