"""
Research Paper Recommender - Core AI Engine
Uses a pretrained SentenceTransformer model for semantic similarity search.

The same model is used in two ways:
1. Local mode: build a FAISS index over uploaded/user-library papers.
2. Online mode: re-rank paper candidates fetched from scholarly APIs.
"""
import hashlib
import os
import pickle
import re
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class PaperRecommender:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize recommender with a pretrained embedding model.

        all-MiniLM-L6-v2 is fast and CPU-friendly. For better paper-specific
        retrieval, you can try: sentence-transformers/allenai-specter
        or all-mpnet-base-v2, but they may be slower/larger.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.papers: List[Dict] = []
        self.index = None
        self.embeddings_cache: List[np.ndarray] = []
        self.paper_keys = set()

    def add_papers(self, papers: List[Dict]) -> int:
        """Add uploaded/local papers to the FAISS index, skipping duplicates."""
        added = 0
        for paper in papers:
            text_to_embed = self._paper_text(paper)
            if not text_to_embed or len(text_to_embed) <= 50:
                continue

            candidate_keys = self._candidate_keys(paper)
            if self.paper_keys.intersection(candidate_keys):
                continue
            paper_key = candidate_keys[0]

            embedding = self.model.encode(text_to_embed[:2000], convert_to_numpy=True)
            self.embeddings_cache.append(embedding)
            self.papers.append(
                {
                    "id": len(self.papers),
                    "title": paper.get("title", "Untitled"),
                    "authors": paper.get("authors", "Unknown"),
                    "year": paper.get("year", ""),
                    "abstract": (paper.get("abstract") or text_to_embed)[:1000],
                    "file_name": paper.get("file_name", ""),
                    "content_hash": paper.get("content_hash", ""),
                    "paper_key": paper_key,
                    "full_text": paper.get("full_text", "")[:2000],
                    "url": paper.get("url", ""),
                    "pdf_url": paper.get("pdf_url", ""),
                    "venue": paper.get("venue", ""),
                    "citation_count": paper.get("citation_count"),
                    "doi": paper.get("doi", ""),
                    "source": paper.get("source", "Local library"),
                }
            )
            self.paper_keys.update(candidate_keys)
            added += 1

        if added > 0 or (self.index is None and self.embeddings_cache):
            self._build_index()
        return added

    def _build_index(self):
        """Build FAISS index for fast local-library similarity search."""
        if not self.embeddings_cache:
            self.index = None
            return

        embeddings_array = np.array(self.embeddings_cache).astype("float32")
        faiss.normalize_L2(embeddings_array)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_array)
        print(f"Index built with {self.index.ntotal} papers")

    def recommend(self, query_text: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Recommend papers from the local uploaded/user library."""
        if not self.index or len(self.papers) == 0:
            return []

        query_embedding = self.model.encode(query_text[:2000], convert_to_numpy=True).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, min(top_k * 3, len(self.papers)))
        recommendations: List[Dict] = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.papers):
                score = self._clamp_score(float(raw_score))
                if score >= min_score:
                    paper = self.papers[idx].copy()
                    paper["similarity_score"] = round(score, 3)
                    paper["relevance"] = self._get_relevance_label(score)
                    recommendations.append(paper)
                    if len(recommendations) >= top_k:
                        break
        return recommendations

    def rerank_candidates(
        self,
        query_text: str,
        candidates: List[Dict],
        top_k: int = 10,
        min_score: float = 0.25,
    ) -> List[Dict]:
        """
        Re-rank online search results using the same pretrained embedding model.

        The API search returns candidate papers. This method converts each
        candidate's title + abstract into an embedding and ranks by cosine
        similarity to the user's research topic.
        """
        if not query_text.strip() or not candidates:
            return []

        valid_candidates: List[Dict] = []
        texts: List[str] = []
        for paper in candidates:
            text_to_embed = self._paper_text(paper)
            if len(text_to_embed) < 20:
                continue
            valid_candidates.append(paper)
            texts.append(text_to_embed[:2500])

        if not valid_candidates:
            return []

        query_embedding = self.model.encode([query_text[:2000]], convert_to_numpy=True).astype("float32")
        paper_embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(paper_embeddings)

        scores = paper_embeddings @ query_embedding[0]
        ranked = sorted(zip(scores, valid_candidates), key=lambda item: float(item[0]), reverse=True)

        recommendations: List[Dict] = []
        for raw_score, paper in ranked:
            score = self._clamp_score(float(raw_score))
            if score < min_score:
                continue
            paper_copy = paper.copy()
            paper_copy["similarity_score"] = round(score, 3)
            paper_copy["relevance"] = self._get_relevance_label(score)
            recommendations.append(paper_copy)
            if len(recommendations) >= top_k:
                break
        return recommendations

    @staticmethod
    def _paper_text(paper: Dict) -> str:
        title = str(paper.get("title", "") or "")
        abstract = str(paper.get("abstract", "") or "")
        full_text = str(paper.get("full_text", "") or paper.get("text", "") or "")
        venue = str(paper.get("venue", "") or "")
        return "\n".join([title, abstract, full_text[:1500], venue]).strip()

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _candidate_keys(self, paper: Dict) -> List[str]:
        """Return all identifiers that should be treated as the same paper."""
        keys: List[str] = []

        for field in ("content_hash", "doi", "url", "pdf_url"):
            value = self._normalize_identifier(paper.get(field, ""))
            if value:
                keys.append(f"{field}:{value}")

        existing_key = self._normalize_identifier(paper.get("paper_key", ""))
        if existing_key:
            keys.append(existing_key)

        title = self._normalize_identifier(paper.get("title", ""))
        if title and title != "untitled":
            keys.append(f"title:{title}")

        file_name = self._normalize_identifier(paper.get("file_name", ""))
        if file_name:
            keys.append(f"file:{file_name}")

        if not keys:
            text_fingerprint = hashlib.sha256(self._paper_text(paper)[:2000].encode("utf-8")).hexdigest()
            keys.append(f"text:{text_fingerprint}")

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(keys))

    def _paper_key(self, paper: Dict) -> str:
        """Return the primary duplicate key for storage."""
        return self._candidate_keys(paper)[0]

    @staticmethod
    def _clamp_score(score: float) -> float:
        # Cosine similarity is not a calibrated probability. We clamp it to
        # 0..1 so the UI can show it as a readable match percentage.
        return max(0.0, min(1.0, score))

    def _get_relevance_label(self, score: float) -> str:
        """Convert similarity score to human-readable label."""
        if score >= 0.7:
            return "Highly Relevant"
        if score >= 0.5:
            return "Relevant"
        if score >= 0.3:
            return "Somewhat Relevant"
        return "Low Relevance"

    def save_index(self, path: str):
        """Save index and papers to disk."""
        if self.index:
            faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({"papers": self.papers, "embeddings_cache": self.embeddings_cache}, f)

    def load_index(self, path: str):
        """Load index and papers from disk."""
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        if os.path.exists(f"{path}.pkl"):
            with open(f"{path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.papers = data["papers"]
                self.embeddings_cache = data["embeddings_cache"]
                self.paper_keys = set()
                for paper in self.papers:
                    self.paper_keys.update(self._candidate_keys(paper))

    def clear(self):
        """Clear all local papers and local index."""
        self.papers = []
        self.embeddings_cache = []
        self.paper_keys = set()
        self.index = None


# Example: Create a sample paper library for demonstration
def create_sample_papers() -> List[Dict]:
    """Create sample research papers for testing the recommender"""
    return [
        {
            "title": "Modeling and Analyzing Taxi Congestion Premium in Congested Cities",
            "authors": "Yuan et al.",
            "year": "2017",
            "abstract": "Traffic congestion is a significant problem in many major cities. Getting stuck in traffic, the mileage per unit time that a taxicab travels will decline significantly. Congestion premium has become an increasingly important income source for taxi drivers. This paper develops a taxi price equilibrium model to investigate the adjustment mechanism of congestion premium on optimizing the taxi driver's income and balancing supply and demand.",
            "file_name": "taxi_congestion_premium.pdf"
        },
        {
            "title": "Extracting Commuter-Specific Destination Hotspots from Trip Destination Data",
            "authors": "Keler et al.",
            "year": "2020",
            "abstract": "Taxi trajectories from urban environments allow inferring various information about transport service qualities and commuter dynamics. This paper compares destination hotspots of boro taxi and Citi Bike users in NYC. The authors introduce a spatiotemporal assigning procedure for areas of influence around static bike sharing stations.",
            "file_name": "commuter_hotspots.pdf"
        },
        {
            "title": "Seeking in Ride-on-Demand Service: A Reinforcement Learning Model with Dynamic Price Prediction",
            "authors": "Guo et al.",
            "year": "2024",
            "abstract": "This paper focuses on the seeking route recommendation problem that aims at increasing driver revenue by recommending profitable seeking routes to drivers of vacant cars with the help of dynamic prices. The authors design a dynamic price prediction model and adopt a reinforcement learning model for route recommendation.",
            "file_name": "ride_on_demand_rl.pdf"
        },
        {
            "title": "Big Data Trip Classification on NYC Taxi and Uber Sensor Network",
            "authors": "Sun et al.",
            "year": "2018",
            "abstract": "This paper uses big data technologies to analyze taxi and Uber trips in New York City. The authors classify regions into three categories based on which service dominates: Yellow taxi, Green taxi, or Uber. Logistic regression achieves over 85% accuracy.",
            "file_name": "big_data_trip_classification.pdf"
        },
        {
            "title": "Changing Demand for New York Yellow Cabs During the COVID-19 Pandemic",
            "authors": "Manley et al.",
            "year": "2021",
            "abstract": "This paper explores the changed spatiotemporal nature of mobility demand during COVID-19. Through comparative analysis of NYC taxi records, the authors observe how relative demand for taxis displaced across land use zones and concentrated during daylight hours.",
            "file_name": "covid_taxi_demand.pdf"
        },
        {
            "title": "Forecasting NYC Yellow Taxi Ridership Decline: A Time Series Analysis",
            "authors": "Singh",
            "year": "2022",
            "abstract": "This study analyzes and forecasts daily passenger counts for NYC yellow taxis during 2017-2019. Using ARIMA models, the analysis reveals strong seasonal patterns with a consistent linear decline of approximately 200 passengers per day.",
            "file_name": "taxi_ridership_forecast.pdf"
        },
        {
            "title": "Effects of Congestion Surcharges: From Ridership to Competition and Safety",
            "authors": "Weber et al.",
            "year": "2023",
            "abstract": "This paper examines the effects of a 2019 congestion surcharge on for-hire-vehicle and taxi usage in NYC. Using difference-in-differences method, the authors find a significant decline in rides originating from the charged area (11%) and a parallel reduction in collisions (5%) and injuries (9%).",
            "file_name": "congestion_surcharges.pdf"
        }
    ]
    return sample_papers