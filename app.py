"""
Research Paper Recommender - Streamlit Web App

Enhanced version:
- Keeps the original local-library PDF recommender.
- Adds online search from Semantic Scholar, OpenAlex, arXiv, and optional
  Google Scholar via SerpApi.
- Re-ranks online candidates with the pretrained embedding model and shows a
  match percentage.
"""
import hashlib
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List

import streamlit as st

try:
    from pdf_processor import PDFProcessor
    PDF_PROCESSOR_IMPORT_ERROR = None
except ImportError as exc:
    PDFProcessor = None
    PDF_PROCESSOR_IMPORT_ERROR = exc

from recommender import PaperRecommender, create_sample_papers
from scholarly_search import SOURCE_LABELS, ScholarlySearchClient


st.set_page_config(page_title="Paper Recommender", page_icon=":books:", layout="wide")

st.title("Research Paper Recommender")
st.markdown(
    "Find relevant papers either from your uploaded/local library or directly "
    "from online scholarly databases."
)

LIBRARY_FILE = "user_library.json"
FLAGGED_FILE = "flagged_papers.json"
CITATIONS_FILE = "my_citations.json"


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------
def save_library():
    """Save current local library to disk."""
    if st.session_state.recommender.papers:
        papers_to_save = []
        for paper in st.session_state.recommender.papers:
            paper_copy = paper.copy()
            paper_copy.pop("embedding", None)
            papers_to_save.append(paper_copy)
        with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(papers_to_save, f, indent=2)


def load_library() -> int:
    """Load local library from disk."""
    if os.path.exists(LIBRARY_FILE):
        try:
            with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
                papers = json.load(f)
            if papers:
                count = st.session_state.recommender.add_papers(papers)
                st.session_state.papers_loaded = count > 0
                return count
        except Exception as exc:
            print(f"Error loading library: {exc}")
    return 0


def save_flagged_papers():
    if st.session_state.flagged_papers:
        with open(FLAGGED_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.flagged_papers, f, indent=2)


def load_flagged_papers() -> List[Dict]:
    if os.path.exists(FLAGGED_FILE):
        try:
            with open(FLAGGED_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_citations():
    if st.session_state.my_citations:
        with open(CITATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.my_citations, f, indent=2)


def load_citations() -> List[Dict]:
    if os.path.exists(CITATIONS_FILE):
        try:
            with open(CITATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "recommender" not in st.session_state:
    st.session_state.recommender = PaperRecommender()
if "papers_loaded" not in st.session_state:
    st.session_state.papers_loaded = False
if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []
if "flagged_papers" not in st.session_state:
    st.session_state.flagged_papers = load_flagged_papers()
if "my_citations" not in st.session_state:
    st.session_state.my_citations = load_citations()
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_search_mode" not in st.session_state:
    st.session_state.last_search_mode = ""
if "sample_papers_loaded" not in st.session_state:
    st.session_state.sample_papers_loaded = False
if "processed_upload_hashes" not in st.session_state:
    st.session_state.processed_upload_hashes = set()
if "library_loaded_from_disk" not in st.session_state:
    st.session_state.library_loaded_from_disk = False

# Load saved library once per Streamlit session. Without this guard, Streamlit
# reruns can repeatedly add the same saved papers to the in-memory index.
if not st.session_state.library_loaded_from_disk:
    loaded_count = load_library()
    st.session_state.library_loaded_from_disk = True
    if loaded_count > 0:
        st.success(f"Loaded {loaded_count} unique papers from your saved local library.")
        st.session_state.sample_papers_loaded = True
        # Re-save after loading so old duplicate entries are cleaned from disk.
        save_library()


# -----------------------------------------------------------------------------
# Small UI helpers
# -----------------------------------------------------------------------------
def stable_key(prefix: str, paper: Dict, index: int) -> str:
    raw = f"{paper.get('source', '')}|{paper.get('external_id', '')}|{paper.get('title', '')}|{index}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def uploaded_file_hash(file) -> str:
    """Return a stable hash for a Streamlit UploadedFile without consuming it."""
    return hashlib.sha256(file.getvalue()).hexdigest()


def add_citation(paper: Dict):
    citation_entry = {
        "title": paper.get("title", "Untitled"),
        "authors": paper.get("authors", "Unknown"),
        "year": paper.get("year", ""),
        "similarity": paper.get("similarity_score", 0),
        "source": paper.get("source", ""),
        "url": paper.get("url", ""),
        "doi": paper.get("doi", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    title = citation_entry["title"]
    if not any(c.get("title") == title for c in st.session_state.my_citations):
        st.session_state.my_citations.append(citation_entry)
        save_citations()
        st.success(f"Added to citations: {title}")
    else:
        st.info(f"Already in citations: {title}")


def flag_paper(paper: Dict):
    title = paper.get("title", "Untitled")
    already_flagged = any(
        isinstance(item, dict) and item.get("title") == title
        for item in st.session_state.flagged_papers
    )
    if not already_flagged:
        st.session_state.flagged_papers.append(
            {
                "title": title,
                "source": paper.get("source", ""),
                "reason": "User flagged as irrelevant",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "query_context": st.session_state.last_query[:300],
            }
        )
        save_flagged_papers()
        st.warning(f"Flagged for review: {title}")
    else:
        st.info("Already flagged this paper.")


def render_paper_card(paper: Dict, index: int):
    score = float(paper.get("similarity_score", 0.0))
    if score >= 0.7:
        score_status = "Strong match"
    elif score >= 0.5:
        score_status = "Good match"
    elif score >= 0.3:
        score_status = "Possible match - verify carefully"
    else:
        score_status = "Weak match - verify carefully"

    title = paper.get("title", "Untitled")
    st.markdown(f"### {index}. {title}")

    meta_col, score_col = st.columns([3, 1])
    with meta_col:
        st.markdown(f"**Authors:** {paper.get('authors', 'Unknown')}")
        st.markdown(f"**Year:** {paper.get('year', '') or 'Unknown'}")
        if paper.get("source"):
            st.markdown(f"**Source:** {paper.get('source')}")
        if paper.get("venue"):
            st.markdown(f"**Venue/categories:** {paper.get('venue')}")
        if paper.get("citation_count") is not None:
            st.markdown(f"**Citations:** {paper.get('citation_count')}")
        if paper.get("doi"):
            st.markdown(f"**DOI:** {paper.get('doi')}")
    with score_col:
        st.metric("Match", f"{score:.0%}")
        st.caption(score_status)

    abstract = paper.get("abstract") or "No abstract/snippet available from this source."
    with st.expander("View abstract/snippet"):
        st.write(abstract)

    link_col, cite_col, flag_col, _ = st.columns([1.2, 1, 1, 3])
    with link_col:
        if paper.get("url"):
            st.link_button("Open paper", paper["url"], use_container_width=True)
        elif paper.get("pdf_url"):
            st.link_button("Open PDF", paper["pdf_url"], use_container_width=True)
    with cite_col:
        if st.button("Cite", key=stable_key("cite", paper, index), use_container_width=True):
            add_citation(paper)
    with flag_col:
        if st.button("Flag", key=stable_key("flag", paper, index), use_container_width=True):
            flag_paper(paper)

    st.divider()


def citation_export_text() -> str:
    lines = []
    for citation in st.session_state.my_citations:
        parts = [
            f"{citation.get('authors', 'Unknown')}.",
            f"({citation.get('year', 'n.d.')}).",
            citation.get("title", "Untitled") + ".",
        ]
        if citation.get("doi"):
            parts.append(f"DOI: {citation['doi']}.")
        if citation.get("url"):
            parts.append(f"URL: {citation['url']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Sidebar: local library management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Local Paper Library")
    st.caption("Local mode uses uploaded PDFs or sample papers. Online mode does not require uploads.")

    if st.button("Load sample papers", use_container_width=True):
        if not st.session_state.sample_papers_loaded:
            with st.spinner("Loading sample papers..."):
                sample_papers = create_sample_papers()
                added = st.session_state.recommender.add_papers(sample_papers)
                st.session_state.papers_loaded = added > 0
                st.session_state.sample_papers_loaded = True
                save_library()
                st.success(f"Loaded {added} sample papers.")
                st.rerun()
        else:
            st.info("Sample papers are already loaded. Clear the library to reload them.")

    st.divider()

    st.subheader("Upload PDFs")

    if PDFProcessor is None:
        st.warning(
            "PDF upload is unavailable because a PDF dependency is missing. "
            "Install it with `python -m pip install pdfplumber PyPDF2`, "
            "or run `python -m pip install -r requirements.txt`. "
            f"Details: {PDF_PROCESSOR_IMPORT_ERROR}"
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=["pdf"],
            accept_multiple_files=True,
            help="Build a personalized local library from PDFs.",
            key="pdf_uploader",
        )

        if uploaded_files:
            st.caption(
                "PDFs are selected but will not be processed until you click "
                "the button below. This prevents Streamlit from adding the same "
                "paper again on every rerun."
            )

            if st.button("Add selected PDFs to library", use_container_width=True):
                pdf_processor = PDFProcessor()
                new_papers = []
                skipped_already_processed = 0

                for file in uploaded_files:
                    file_bytes = file.getvalue()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()

                    if file_hash in st.session_state.processed_upload_hashes:
                        skipped_already_processed += 1
                        continue

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file_bytes)
                        tmp_path = tmp_file.name

                    try:
                        with st.spinner(f"Processing {file.name}..."):
                            paper_data = pdf_processor.process_paper(tmp_path)
                            if paper_data:
                                paper_data["title"] = file.name.replace(".pdf", "").replace("_", " ").title()
                                paper_data["file_name"] = file.name
                                paper_data["content_hash"] = file_hash
                                paper_data["source"] = "Local upload"
                                new_papers.append(paper_data)
                                st.session_state.processed_upload_hashes.add(file_hash)
                    finally:
                        os.unlink(tmp_path)

                added = st.session_state.recommender.add_papers(new_papers) if new_papers else 0
                st.session_state.papers_loaded = len(st.session_state.recommender.papers) > 0

                if added > 0:
                    save_library()
                    st.success(f"Added {added} new unique paper(s).")
                    st.info("You can remove the selected file from the uploader using the X button.")
                elif skipped_already_processed > 0:
                    st.info("This selected PDF was already processed in this session, so it was skipped.")
                else:
                    st.info("No new unique papers were added. This file may already be in the library.")

    st.divider()

    st.subheader("Stats")
    st.metric("Local papers", len(st.session_state.recommender.papers))
    st.metric("Flagged", len(st.session_state.flagged_papers))
    st.metric("Citations", len(st.session_state.my_citations))

    if st.session_state.my_citations:
        st.download_button(
            label="Export citations",
            data=citation_export_text(),
            file_name=f"my_citations_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()

    if st.button("Clear local library and feedback", use_container_width=True):
        st.session_state.recommender.clear()
        st.session_state.papers_loaded = False
        st.session_state.recommendation_history = []
        st.session_state.flagged_papers = []
        st.session_state.my_citations = []
        st.session_state.sample_papers_loaded = False
        st.session_state.processed_upload_hashes = set()
        st.session_state.library_loaded_from_disk = True
        for file_name in [LIBRARY_FILE, FLAGGED_FILE, CITATIONS_FILE]:
            if os.path.exists(file_name):
                os.remove(file_name)
        st.success("Local library and feedback cleared.")
        st.rerun()


# -----------------------------------------------------------------------------
# Main search area
# -----------------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.header("Search")

    search_mode = st.radio(
        "Where should the app search?",
        ["Online scholarly databases", "My uploaded/local library"],
        horizontal=False,
    )

    query_text = st.text_area(
        "Research topic, keywords, or paragraph",
        placeholder=(
            "Example: graph neural networks for fake news detection using "
            "social context and propagation patterns"
        ),
        height=180,
    )

    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        top_k = st.selectbox("Number of papers to show", [3, 5, 10, 15], index=1)
    with settings_col2:
        default_threshold = 0.25 if search_mode == "Online scholarly databases" else 0.3
        min_score = st.slider(
            "Minimum match score",
            0.0,
            1.0,
            default_threshold,
            0.05,
            help="This is semantic similarity, not a true calibrated probability.",
        )

    selected_source_keys: List[str] = []
    candidate_limit = 30
    year_from = None
    year_to = None

    if search_mode == "Online scholarly databases":
        st.info(
            "Online mode fetches paper candidates from scholarly APIs, then your "
            "pretrained embedding model re-ranks them by match score."
        )
        label_to_key = {label: key for key, label in SOURCE_LABELS.items()}
        source_labels = [
            "Semantic Scholar",
            "OpenAlex",
            "arXiv",
            "Google Scholar (SerpApi)",
        ]
        selected_labels = st.multiselect(
            "Sources",
            source_labels,
            default=["Semantic Scholar", "OpenAlex", "arXiv"],
            help="Google Scholar requires a SERPAPI_API_KEY environment variable.",
        )
        selected_source_keys = [label_to_key[label] for label in selected_labels]
        candidate_limit = st.slider(
            "Candidate papers per source",
            5,
            100,
            30,
            5,
            help="More candidates can improve recall but make search slower.",
        )
        use_year_filter = st.checkbox("Limit publication years")
        if use_year_filter:
            year_col1, year_col2 = st.columns(2)
            with year_col1:
                year_from = st.number_input("From year", min_value=1900, max_value=2100, value=2020)
            with year_col2:
                year_to = st.number_input("To year", min_value=1900, max_value=2100, value=datetime.now().year)
            if year_from and year_to and year_from > year_to:
                st.warning("From year should be less than or equal to To year.")

    if st.button("Find relevant papers", type="primary", use_container_width=True):
        query = query_text.strip()
        if not query:
            st.error("Please enter a research topic, keywords, or paragraph.")
        elif search_mode == "My uploaded/local library" and not st.session_state.papers_loaded:
            st.error("Please upload papers or load sample papers for local-library search.")
        elif search_mode == "Online scholarly databases" and not selected_source_keys:
            st.error("Please select at least one online source.")
        else:
            st.session_state.last_query = query
            st.session_state.last_search_mode = search_mode
            if search_mode == "My uploaded/local library":
                with st.spinner("Searching your local library..."):
                    recommendations = st.session_state.recommender.recommend(
                        query,
                        top_k=top_k,
                        min_score=min_score,
                    )
            else:
                with st.spinner("Searching online sources and re-ranking candidates..."):
                    client = ScholarlySearchClient()
                    candidates = client.search(
                        query,
                        limit=candidate_limit,
                        sources=selected_source_keys,
                        year_from=int(year_from) if year_from else None,
                        year_to=int(year_to) if year_to else None,
                    )
                    recommendations = st.session_state.recommender.rerank_candidates(
                        query,
                        candidates,
                        top_k=top_k,
                        min_score=min_score,
                    )
                    if client.last_errors:
                        st.warning("Some sources could not be searched: " + "; ".join(client.last_errors))
                    st.caption(f"Fetched {len(candidates)} unique candidate papers before re-ranking.")

            st.session_state.recommendation_history = recommendations
            if recommendations:
                st.success(f"Found {len(recommendations)} relevant papers.")
                if recommendations[0].get("similarity_score", 0) < 0.5:
                    st.warning(
                        "No strong match was found. Try more specific terms, lower the "
                        "threshold, or fetch more online candidates."
                    )
            else:
                st.warning(
                    f"No papers found above {min_score:.0%}. Try lowering the match score "
                    "or changing the query."
                )

with right:
    st.header("Recommended Papers")
    st.caption("Ranked by semantic match to your topic. Verify all papers before citing.")

    if st.session_state.recommendation_history:
        st.caption(f"Last search mode: {st.session_state.last_search_mode}")
        for idx, paper in enumerate(st.session_state.recommendation_history, 1):
            render_paper_card(paper, idx)
    else:
        st.info("Enter a topic and click Find relevant papers to see recommendations.")

    if st.session_state.flagged_papers:
        with st.expander("Flagged papers"):
            for flagged in st.session_state.flagged_papers:
                st.write(
                    f"- {flagged.get('title', 'Untitled')} "
                    f"({flagged.get('source', 'unknown source')}, {flagged.get('timestamp', '')})"
                )


# -----------------------------------------------------------------------------
# Explanation / hackathon notes
# -----------------------------------------------------------------------------
st.divider()
st.markdown(
    """
### How this version works

1. **Local mode:** uploaded PDFs are converted to text, embedded with the pretrained SentenceTransformer model, indexed with FAISS, and searched locally.
2. **Online mode:** the app calls scholarly APIs, collects candidate papers, embeds each candidate's title/abstract, and re-ranks them against the user's topic.
3. **Match score:** the displayed percentage is cosine similarity converted to a readable 0-100% score. Treat it as a relevance score, not a mathematically calibrated probability.

**Recommended hackathon setup:** use Semantic Scholar + OpenAlex + arXiv by default. Use Google Scholar only through SerpApi by setting `SERPAPI_API_KEY`; avoid direct Google Scholar scraping.
"""
)

if st.session_state.my_citations:
    with st.expander(f"My citations ({len(st.session_state.my_citations)} papers)"):
        for cited in st.session_state.my_citations:
            st.write(
                f"- {cited.get('title', 'Untitled')} ({cited.get('year', '')}) "
                f"- {cited.get('source', '')} - cited on {cited.get('timestamp', '')}"
            )

# -----------------------------------------------------------------------------
# Entry point for Hugging Face Spaces
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from streamlit.web import cli as stcli
    
    # Ensure the app runs on the correct port
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())
