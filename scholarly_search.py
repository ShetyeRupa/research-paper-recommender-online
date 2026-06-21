"""
Online scholarly search connectors for the Research Paper Recommender.

Supported sources:
- Semantic Scholar Academic Graph API (free; optional API key for higher limits)
- OpenAlex Works API (free; optional OPENALEX_EMAIL for polite pool)
- arXiv API (free; best for CS, physics, math, quantitative biology, etc.)
- Google Scholar through SerpApi (optional paid/free-tier key; Google Scholar has no official public API)
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import requests


SEMANTIC_SCHOLAR_FIELDS = (
    "title,authors,year,abstract,url,venue,citationCount,externalIds,"
    "openAccessPdf,publicationDate"
)

SOURCE_LABELS = {
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
    "arxiv": "arXiv",
    "google_scholar_serpapi": "Google Scholar (SerpApi)",
}


class ScholarlySearchClient:
    """Fetch paper candidates from public scholarly APIs."""

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.last_errors: List[str] = []
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self.openalex_email = os.getenv("OPENALEX_EMAIL", "").strip()
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY", "").strip()

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "ResearchPaperRecommenderHackathon/1.0 "
                    "(https://github.com/; contact: set OPENALEX_EMAIL env var)"
                )
            }
        )
        if self.semantic_scholar_api_key:
            self.session.headers.update({"x-api-key": self.semantic_scholar_api_key})

    def search(
        self,
        query: str,
        limit: int = 30,
        sources: Sequence[str] = ("semantic_scholar", "openalex", "arxiv"),
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[Dict]:
        """Search one or more scholarly sources and return de-duplicated paper dicts."""
        self.last_errors = []
        query = (query or "").strip()
        if not query:
            return []

        limit = max(1, min(int(limit), 100))
        sources = [s for s in sources if s in SOURCE_LABELS]
        if not sources:
            sources = ["semantic_scholar"]

        all_papers: List[Dict] = []
        for source in sources:
            try:
                if source == "semantic_scholar":
                    papers = self.search_semantic_scholar(query, limit, year_from, year_to)
                elif source == "openalex":
                    papers = self.search_openalex(query, limit, year_from, year_to)
                elif source == "arxiv":
                    papers = self.search_arxiv(query, min(limit, 50), year_from, year_to)
                elif source == "google_scholar_serpapi":
                    papers = self.search_google_scholar_serpapi(query, min(limit, 20), year_from, year_to)
                else:
                    papers = []
                all_papers.extend(papers)
            except Exception as exc:  # Keep app running when a source is down/rate-limited.
                self.last_errors.append(f"{SOURCE_LABELS.get(source, source)}: {exc}")

        return self._deduplicate(all_papers)

    def search_semantic_scholar(
        self,
        query: str,
        limit: int,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict]:
        params = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        if year_from or year_to:
            start = year_from if year_from else ""
            end = year_to if year_to else ""
            params["year"] = f"{start}-{end}"

        response = self.session.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            timeout=self.timeout,
        )
        if response.status_code == 429:
            raise RuntimeError("rate limit reached; add SEMANTIC_SCHOLAR_API_KEY or try fewer results")
        response.raise_for_status()

        papers: List[Dict] = []
        for item in response.json().get("data", []) or []:
            title = clean_text(item.get("title", ""))
            if not title:
                continue
            authors = ", ".join(
                clean_text(author.get("name", ""))
                for author in (item.get("authors") or [])[:8]
                if author.get("name")
            )
            external_ids = item.get("externalIds") or {}
            open_access_pdf = item.get("openAccessPdf") or {}
            papers.append(
                {
                    "title": title,
                    "authors": authors or "Unknown",
                    "year": item.get("year") or "",
                    "abstract": clean_text(item.get("abstract", "")),
                    "url": item.get("url") or open_access_pdf.get("url") or "",
                    "pdf_url": open_access_pdf.get("url") or "",
                    "venue": clean_text(item.get("venue", "")),
                    "citation_count": item.get("citationCount"),
                    "doi": external_ids.get("DOI", ""),
                    "source": "Semantic Scholar",
                    "external_id": item.get("paperId", ""),
                }
            )
        return papers

    def search_openalex(
        self,
        query: str,
        limit: int,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict]:
        params = {
            "search": query,
            "per-page": max(1, min(limit, 100)),
            "select": (
                "id,doi,title,display_name,publication_year,authorships,"
                "abstract_inverted_index,cited_by_count,primary_location,open_access"
            ),
        }
        if self.openalex_email:
            params["mailto"] = self.openalex_email

        filters: List[str] = []
        if year_from:
            filters.append(f"from_publication_date:{int(year_from)}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{int(year_to)}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        response = self.session.get(
            "https://api.openalex.org/works",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        papers: List[Dict] = []
        for item in response.json().get("results", []) or []:
            title = clean_text(item.get("title") or item.get("display_name") or "")
            if not title:
                continue
            authors = ", ".join(
                clean_text((authorship.get("author") or {}).get("display_name", ""))
                for authorship in (item.get("authorships") or [])[:8]
                if (authorship.get("author") or {}).get("display_name")
            )
            primary_location = item.get("primary_location") or {}
            source = primary_location.get("source") or {}
            open_access = item.get("open_access") or {}
            papers.append(
                {
                    "title": title,
                    "authors": authors or "Unknown",
                    "year": item.get("publication_year") or "",
                    "abstract": openalex_abstract(item.get("abstract_inverted_index")),
                    "url": primary_location.get("landing_page_url") or item.get("doi") or item.get("id", ""),
                    "pdf_url": open_access.get("oa_url") or "",
                    "venue": clean_text(source.get("display_name", "")),
                    "citation_count": item.get("cited_by_count"),
                    "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
                    "source": "OpenAlex",
                    "external_id": item.get("id", ""),
                }
            )
        return papers

    def search_arxiv(
        self,
        query: str,
        limit: int,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict]:
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max(1, min(limit, 50)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = self.session.get(
            "https://export.arxiv.org/api/query",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        papers: List[Dict] = []
        for entry in root.findall("atom:entry", ns):
            title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
            abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = int(published[:4]) if published[:4].isdigit() else None
            if year_from and year and year < int(year_from):
                continue
            if year_to and year and year > int(year_to):
                continue
            authors = ", ".join(
                clean_text(author.findtext("atom:name", default="", namespaces=ns))
                for author in entry.findall("atom:author", ns)[:8]
            )
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            categories = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ns)]
            papers.append(
                {
                    "title": title,
                    "authors": authors or "Unknown",
                    "year": year or "",
                    "abstract": abstract,
                    "url": entry_id,
                    "pdf_url": pdf_url,
                    "venue": ", ".join([c for c in categories if c]),
                    "citation_count": None,
                    "doi": "",
                    "source": "arXiv",
                    "external_id": entry_id.rsplit("/", 1)[-1] if entry_id else "",
                }
            )
        return papers

    def search_google_scholar_serpapi(
        self,
        query: str,
        limit: int,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict]:
        if not self.serpapi_api_key:
            raise RuntimeError("SERPAPI_API_KEY is not set")

        params = {
            "engine": "google_scholar",
            "q": query,
            "num": max(1, min(limit, 20)),
            "api_key": self.serpapi_api_key,
            "output": "json",
        }
        if year_from:
            params["as_ylo"] = int(year_from)
        if year_to:
            params["as_yhi"] = int(year_to)

        response = self.session.get("https://serpapi.com/search", params=params, timeout=self.timeout)
        response.raise_for_status()
        results = response.json().get("organic_results", []) or []

        papers: List[Dict] = []
        for item in results:
            publication_info = item.get("publication_info") or {}
            inline_links = item.get("inline_links") or {}
            cited_by = inline_links.get("cited_by") or {}
            summary = clean_text(publication_info.get("summary", ""))
            papers.append(
                {
                    "title": clean_text(item.get("title", "")),
                    "authors": parse_google_scholar_authors(publication_info) or "Unknown",
                    "year": parse_year(summary),
                    "abstract": clean_text(item.get("snippet", "")),
                    "url": item.get("link", ""),
                    "pdf_url": "",
                    "venue": summary,
                    "citation_count": cited_by.get("total"),
                    "doi": "",
                    "source": "Google Scholar (SerpApi)",
                    "external_id": item.get("result_id", ""),
                }
            )
        return [paper for paper in papers if paper.get("title")]

    @staticmethod
    def _deduplicate(papers: List[Dict]) -> List[Dict]:
        seen = set()
        unique: List[Dict] = []
        for paper in papers:
            key = paper_key(paper)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(paper)
        return unique


def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_year(text: str) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", text or "")
    return match.group(0) if match else ""


def parse_google_scholar_authors(publication_info: Dict) -> str:
    authors = publication_info.get("authors") or []
    if isinstance(authors, list) and authors:
        names = [clean_text(author.get("name", "")) for author in authors if author.get("name")]
        if names:
            return ", ".join(names[:8])
    summary = clean_text(publication_info.get("summary", ""))
    if " - " in summary:
        return summary.split(" - ", 1)[0]
    return ""


def openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index:
        return ""
    positioned_words: List[Tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            positioned_words.append((int(pos), word))
    positioned_words.sort(key=lambda item: item[0])
    return clean_text(" ".join(word for _, word in positioned_words))


def paper_key(paper: Dict) -> str:
    doi = clean_text(paper.get("doi", "")).lower()
    if doi:
        return f"doi:{doi}"
    title = clean_text(paper.get("title", "")).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    return f"title:{title}" if title else ""
