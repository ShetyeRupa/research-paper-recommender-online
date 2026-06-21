"""
PDF Processor - Extracts text from research papers
"""
import re
from typing import List, Dict
import pdfplumber
import PyPDF2

class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_text_pdfplumber(self, file_path: str) -> List[Dict]:
        """Extract text from PDF using pdfplumber"""
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        pages.append({
                            "page": page_num,
                            "text": text,
                            "char_count": len(text)
                        })
            return pages
        except Exception as e:
            print(f"Error with pdfplumber: {e}")
            return []
    
    def extract_text_pypdf2(self, file_path: str) -> List[Dict]:
        """Extract text from PDF using PyPDF2"""
        pages = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        pages.append({
                            "page": page_num,
                            "text": text,
                            "char_count": len(text)
                        })
            return pages
        except Exception as e:
            print(f"Error with PyPDF2: {e}")
            return []
    
    def extract_text(self, file_path: str) -> List[Dict]:
        """Extract text from PDF"""
        pages = self.extract_text_pdfplumber(file_path)
        if not pages or all(p['char_count'] < 100 for p in pages):
            pages = self.extract_text_pypdf2(file_path)
        return pages
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract academic paper sections"""
        sections = {
            "abstract": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": ""
        }
        
        # Improved patterns with better group handling
        patterns = {
            "abstract": r'(?i)(?:abstract|summary)[\s\n:]+(.+?)(?=\n\n|\n[A-Z]|$)',
            "introduction": r'(?i)(?:introduction|background)[\s\n:]+(.+?)(?=\n\n|\n[A-Z]|$)',
            "methods": r'(?i)(?:methods?|methodology|experimental|experiments?)[\s\n:]+(.+?)(?=\n\n|\n[A-Z]|$)',
            "results": r'(?i)results[\s\n:]+(.+?)(?=\n\n|\n[A-Z]|$)',
            "discussion": r'(?i)discussion[\s\n:]+(.+?)(?=\n\n|\n[A-Z]|$)',
            "conclusion": r'(?i)(?:conclusion|conclusions)[\s\n:]+(.+?)(?=\n\n|\Z)'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match and match.group(1):
                sections[section] = match.group(1).strip()[:1000]
        
        return sections
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks"""
        if not text:
            return []
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 100:
                chunks.append(chunk)
        return chunks
    
    def process_paper(self, file_path: str) -> Dict:
        """Process a single paper"""
        pages = self.extract_text(file_path)
        if not pages:
            return None
        
        full_text = " ".join([p['text'] for p in pages])
        
        # Limit text to avoid memory issues
        if len(full_text) > 50000:
            full_text = full_text[:50000]
        
        sections = self.extract_sections(full_text)
        
        return {
            "file_name": file_path.split("/")[-1],
            "total_pages": len(pages),
            "full_text": full_text,
            "abstract": sections.get("abstract", "")[:2000],
            "introduction": sections.get("introduction", "")[:2000],
            "metadata": {
                "page_count": len(pages), 
                "has_abstract": bool(sections.get("abstract"))
            }
        }