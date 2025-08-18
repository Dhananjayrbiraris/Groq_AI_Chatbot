# search.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import Optional, Dict, Any
from normalize import normalize_filters



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        self.ids: list[int] = []

    def add(self, vecs: np.ndarray, ids: list[int]):
        self.index.add(vecs.astype(np.float32))
        self.ids.extend(ids)

    def search(self, qvec: np.ndarray, k: int = 20):
        D, I = self.index.search(qvec.astype(np.float32), k)
        return D, I

class Retriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.doc_matrix = None

    def _row_text(self, row: pd.Series) -> str:
        fields = [
            row.get("Job Title",""),
            row.get("Parent Department",""),
            row.get("Seniority",""),
            row.get("Industry",""),
            row.get("Company Name",""),
            row.get("Location","")
        ]
        return " | ".join(map(str, fields))

    def build(self):
        texts = [self._row_text(r) for _, r in self.df.iterrows()]
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
        self.doc_matrix = embs
        self.index = FaissIndex(dim=embs.shape[1])
        self.index.add(embs, list(range(len(texts))))
        return self

    def filter_df(self, Department=None, Seniority=None, JobTitle=None, LocationLike=None,
              Industry=None, Company=None) -> pd.DataFrame:
        df = self.df
        def contains(col, val):
            return df[col].astype(str).str.contains(val, case=False, na=False, regex=False)

        if Department:
            df = df[contains("Parent Department", Department)]

        def apply_filters(df: pd.DataFrame, filters: dict):
            filters = normalize_filters(filters)

            if filters.get("Department"):
                df = df[df["Department"].str.contains(filters["Department"], case=False, na=False)]

            if filters.get("Seniority"):
                seniorities = filters["Seniority"]
                if isinstance(seniorities, list):
                    mask = False
                    for s in seniorities:
                        mask = mask | df["Seniority"].str.contains(s, case=False, na=False)
                    df = df[mask]
                else:
                    df = df[df["Seniority"].str.contains(seniorities, case=False, na=False)]

            if filters.get("LocationLike"):
                df = df[df["Location"].str.contains(filters["LocationLike"], case=False, na=False)]

            return df
        
        df = self.df
        def contains(col, val):
            return df[col].astype(str).str.contains(val, case=False, na=False, regex=False)

        if Department:
            df = df[contains("Parent Department", Department)]
        if Seniority:
            df = df[contains("Seniority", Seniority)]
        if JobTitle:
            df = df[contains("Job Title", JobTitle)]
        if LocationLike:
            # match across Location, City, State, Country-like fields
            mask = (
                df["Location"].astype(str).str.contains(LocationLike, case=False, na=False, regex=False) |
                df["Contact City"].astype(str).str.contains(LocationLike, case=False, na=False, regex=False) |
                df["Contact State"].astype(str).str.contains(LocationLike, case=False, na=False, regex=False)
            )
            df = df[mask]
        if Industry:
            df = df[contains("Industry", Industry)]
        if Company:
            df = df[contains("Company Name", Company)]
        return df

    def semantic_rerank(self, df: pd.DataFrame, query: str, top_k: int = 20) -> pd.DataFrame:
        if df.empty:
            return df
        qvec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        # search full corpus, then keep only rows in df
        D, I = self.index.search(qvec, k=min(top_k*5, len(self.df)))
        idxs = [i for i in I[0].tolist() if i in df.index.tolist()]
        scores = D[0][:len(idxs)]
        out = df.loc[idxs].copy()
        out["similarity_score"] = scores
        out = out.sort_values("similarity_score", ascending=False)
        return out.head(top_k)
