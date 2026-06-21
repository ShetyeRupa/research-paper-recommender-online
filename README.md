
---
title: Research Paper Recommender Online
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_port: 7860
pinned: false
license: mit
tags:
  - research
  - nlp
  - semantic-search
  - paper-recommender
  - streamlit
  - sentence-transformers
  - faiss
  - academic
---

# 📚 Research Paper Recommender Online

> *AI-powered semantic search engine that helps researchers find the most relevant papers from their library or online databases*

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/ShetyeRupa/research-paper-recommender-online)
[![GitHub](https://img.shields.io/badge/🐙-GitHub%20Repo-black)](https://github.com/ShetyeRupa/research-paper-recommender-online)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

---

## 🚀 Live Demo

**Try it now:** [https://huggingface.co/spaces/ShetyeRupa/research-paper-recommender-online](https://huggingface.co/spaces/ShetyeRupa/research-paper-recommender-online)

---

## 📖 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Deployment](#-deployment)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Responsible AI](#-responsible-ai)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Problem Statement

Graduate students and researchers waste hours searching their own paper libraries for relevant citations. They know a paper exists. They have read it. But when writing mid-paragraph, they cannot retrieve it. This leads to:

- ⏱️ **Hours lost** searching through folders and reference managers
- 📝 **Missed citations** that weaken literature reviews
- ❌ **Mentor rejections** due to incomplete references
- 🎓 **Delayed graduation** from repeated revisions

**Our solution:** An intelligent paper recommender that reads your writing and instantly surfaces the most semantically relevant papers from your library or online databases.

---

## ✨ Features

| Feature | Description |
|:---|:---|
| 🔍 **Semantic Search** | Uses Sentence-BERT embeddings to understand meaning, not just keywords |
| 📄 **PDF Processing** | Extract text from research papers automatically |
| 🌐 **Online Search** | Search Semantic Scholar, OpenAlex, arXiv, and Google Scholar |
| 🎯 **Relevance Scoring** | Each recommendation includes 0-100% similarity score |
| 👩‍💻 **Human-in-the-Loop** | Users control which papers to cite |
| 🚩 **Flag Bad Recommendations** | Provide feedback to improve the system |
| ⚠️ **Confidence Threshold** | Silent failure mitigation when no strong matches found |
| 📋 **Export Citations** | Download cited papers as formatted text |
| 💾 **Persistent Library** | Your uploaded papers saved between sessions |

---

## 🏗️ Architecture

```
[User Query] → [Embedding Generation] → [Similarity Search]
         ↓                                    ↓
    [Semantic Scholar]                  [FAISS Index]
    [OpenAlex]                          [Local Library]
    [arXiv]
    [Google Scholar]

[PDF Upload] → [Text Extraction] → [Section Segmentation] → [Library Cache]
```

### AI Model Selection

| Component | Choice | Rationale |
|:---|:---|:---|
| **Embedding Model** | `all-MiniLM-L6-v2` (Sentence-BERT) | 384-dim, CPU-friendly, 0.5s per 100 papers |
| **Similarity Search** | FAISS (IndexFlatIP) | Cosine similarity after L2 normalization |
| **PDF Extraction** | pdfplumber + PyPDF2 | Dual fallback for maximum compatibility |

### Why Semantic Similarity Over Keyword Search?

| Approach | Limitation | Our Solution |
|:---|:---|:---|
| **BM25 / TF-IDF** | Fails on synonymy ("attention mechanism" ≈ "transformer self-attention") | Dense embeddings map semantically similar texts close together |
| **Keyword Matching** | Cannot distinguish methods vs. discussion relevance | Section-aware embedding generation |

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---|:---|
| **Frontend** | Streamlit |
| **AI/ML** | Sentence-BERT, FAISS, Transformers |
| **PDF Processing** | pdfplumber, PyPDF2 |
| **Data Processing** | NumPy, Pandas |
| **Deployment** | Hugging Face Spaces, Docker |
| **Version Control** | GitHub |

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda

### Local Setup

```bash
# Clone the repository
git clone https://github.com/ShetyeRupa/research-paper-recommender-online.git
cd research-paper-recommender

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Using Conda

```bash
# Create conda environment
conda create -n paper_recommender python=3.11 -y
conda activate paper_recommender

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Docker Setup

```bash
# Build Docker image
docker build -t research-paper-recommender .

# Run Docker container
docker run -p 7860:7860 research-paper-recommender
```

---

## 🚀 Deployment

### Hugging Face Spaces

This app is deployed on Hugging Face Spaces using Docker.

**Quick Deploy:**

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Docker** as SDK
3. Clone the Space locally
4. Copy all project files
5. Push to the Space

```bash
# Clone your Space
git clone https://huggingface.co/spaces/ShetyeRupa/research-paper-recommender-online
cd research-paper-recommender-online

# Copy project files
cp -r /path/to/your/project/* .

# Push to Hugging Face
git add .
git commit -m "Deploy Research Paper Recommender"
git push
```

**Required Files:**
- `Dockerfile` - Container configuration
- `space.yaml` - Space configuration
- `requirements.txt` - Python dependencies
- `app.py` - Main application
- `recommender.py` - Core AI engine
- `pdf_processor.py` - PDF processing
- `scholarly_search.py` - Online search connectors

**Environment Variables (Secrets):**

Add these in your Space settings:

| Variable | Description | Required |
|:---|:---|:---|
| `SERPAPI_API_KEY` | Google Scholar search | Optional |
| `SEMANTIC_SCHOLAR_API_KEY` | Higher rate limits | Optional |
| `OPENALEX_EMAIL` | Polite pool access | Optional |

### GitHub Repository

**Push to GitHub:**

```bash
git remote add origin https://github.com/ShetyeRupa/research-paper-recommender-online.git
git branch -M main
git push -u origin main
```

**Repository:** [https://github.com/ShetyeRupa/research-paper-recommender-online](https://github.com/ShetyeRupa/research-paper-recommender-online)

---

## 🧪 Example Queries

Try these research topics:

```
How does congestion pricing affect taxi ridership and traffic safety in urban areas?
```

```
What are the best practices for extracting destination hotspots from taxi trajectory data?
```

```
How can reinforcement learning be used to optimize driver revenue in ride-on-demand services?
```

```
Graph neural networks for fake news detection using social context and propagation patterns
```

```
Transformer-based models for natural language understanding in low-resource languages
```

---

## 📊 Evaluation Metrics

| Metric | Target | Achieved |
|:---|:---|:---|
| **Recall@5** | >0.70 | ✓ Validated on held-out test set |
| **Precision@5** | >0.65 | ✓ Validated on held-out test set |
| **Latency (p95)** | <3 seconds | ~1.5s per query on CPU |

---

## 🤖 Responsible AI

Every AI system has risks. Our mitigations:

| Risk | Mitigation |
|:---|:---|
| **Over-reliance** | Warning message: "AI may miss papers. Always verify." |
| **Missing relevant papers** | Similarity scores displayed; user can search manually |
| **Domain drift (new terminology)** | Weekly re-embedding + manual keyword override |
| **Silent failure** | Confidence threshold (below 0.5 triggers "No strong matches found") |
| **User bias** | Flagging system collects feedback for improvement |

### Human-in-the-Loop Design

- ✅ User decides which papers to cite
- ✅ Similarity scores displayed with every recommendation
- ✅ Confidence threshold filter (adjustable by user)
- ✅ Flag bad recommendations to improve the system
- ✅ Export citations for verification

---

## 📁 Project Structure

```
research-paper-recommender/
├── app.py                 # Streamlit web application
├── recommender.py         # Core AI engine (Sentence-BERT + FAISS)
├── pdf_processor.py       # PDF text extraction and section parsing
├── scholarly_search.py    # Online scholarly search connectors
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── space.yaml             # Hugging Face Space configuration
├── README.md              # This file
└── LICENSE                # MIT License
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|:---|:---|
| `ModuleNotFoundError: No module named 'faiss'` | Install: `pip install faiss-cpu` or `conda install -c conda-forge faiss-cpu` |
| `ModuleNotFoundError: No module named 'torch'` | Install: `pip install torch` |
| PDF text extraction empty | Ensure PDF is text-based (not scanned image) |
| Slow performance on first run | Model downloads on first load; subsequent runs are cached |
| API rate limits | Add API keys in environment variables |
| Port already in use | Change port: `streamlit run app.py --server.port=8502` |

---

## 🗺️ Roadmap

- [ ] Integration with Zotero/Mendeley APIs
- [ ] Multi-language paper support
- [ ] Collaborative library sharing
- [ ] Citation graph visualization
- [ ] Browser extension for direct PDF import
- [ ] Recommendation history dashboard
- [ ] PDF annotation and highlighting
- [ ] Team collaboration features

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Team

| Role | Name | GitHub |
|:---|:---|:---|
| **AI Engineer** | Rupali Shetye | [@ShetyeRupa](https://github.com/ShetyeRupa) |
| **AI Engineer** | Kartavya Mandora | [@Kartavya1905](https://github.com/Kartavya1905) |

**Track:** Graduate

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **USAII®** for organizing the Global AI Hackathon 2026
- **Sentence-BERT** team for the embedding model
- **FAISS** team for efficient similarity search
- **Streamlit** for making ML app deployment seamless
- **Hugging Face** for hosting the Space
- **Semantic Scholar, OpenAlex, arXiv** for providing free APIs

---

## 📧 Contact

- **Project Repository:** [https://github.com/ShetyeRupa/research-paper-recommender-online](https://github.com/ShetyeRupa/research-paper-recommender-online)
- **Live Demo:** [https://huggingface.co/spaces/ShetyeRupa/research-paper-recommender-online](https://huggingface.co/spaces/ShetyeRupa/research-paper-recommender-online)

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub and share it with your network!

---

**Built with ❤️ for Global AI Hackathon 2026**

---

## Online Scholarly Search

This version can search beyond the local uploaded-PDF library.

### Supported Online Sources

- **Semantic Scholar** - Academic Graph API (free; optional API key for higher limits)
- **OpenAlex** - Works API (free; optional OPENALEX_EMAIL for polite pool)
- **arXiv** - API (free; best for CS, physics, math, quantitative biology, etc.)
- **Google Scholar** - Through SerpApi (optional paid/free-tier key)

The app does **not** scrape Google Scholar directly. To use Google Scholar results legally and reliably, create a SerpApi key.

### Online Search Pipeline

```text
Research topic → Scholarly API candidate search → Title/abstract embeddings → Cosine similarity re-ranking → Top papers with match scores
```

The match score shown in the UI is a semantic similarity percentage, not a calibrated statistical probability.

---

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@misc{research-paper-recommender-2026,
  author = {Shetye, Rupali and Kartavya},
  title = {Research Paper Recommender: AI-Powered Semantic Search},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ShetyeRupa/research-paper-recommender-online}
}
```
