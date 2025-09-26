# backend/chunking.py
# Enhanced chunking strategies with better text processing and validation

from typing import List, Dict, Any, Optional, Tuple
import hashlib
import re
import nltk
from dataclasses import dataclass
from collections import defaultdict

# Ensure required NLTK resources are available
NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords"]
for resource in NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource in ["punkt", "punkt_tab"] else f"corpora/{resource}")
    except LookupError:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

@dataclass
class ChunkConfig:
    """Configuration for chunking parameters"""
    chunk_size: int = 512
    overlap: int = 64
    min_chunk_size: int = 50
    max_chunk_size: int = 1000
    sentence_split: bool = True
    preserve_structure: bool = True
    remove_duplicates: bool = True

class TextPreprocessor:
    """Enhanced text preprocessing utilities"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\:\;\-\(\)]', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([\.!\?])', r'\1', text)
        text = re.sub(r'([\.!\?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def is_meaningful_text(self, text: str, min_words: int = 3) -> bool:
        """Check if text chunk contains meaningful content"""
        if not text or len(text.strip()) < 20:
            return False
        
        words = text.split()
        if len(words) < min_words:
            return False
        
        # Check if it's mostly non-alphabetic (like navigation, numbers, etc.)
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.5:
            return False
        
        # Check if it's mostly stop words
        meaningful_words = [w.lower() for w in words if w.lower() not in self.stop_words and w.isalpha()]
        if len(meaningful_words) / len(words) < 0.3:
            return False
        
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """Better token estimation (closer to actual tokenization)"""
        # Rough approximation: 1 token ≈ 0.75 words for English
        words = len(text.split())
        return int(words * 1.33)

def _make_id(text: str, additional_info: str = "") -> str:
    """Generate consistent chunk ID with optional additional context"""
    content = f"{text}_{additional_info}" if additional_info else text
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

class ChunkValidator:
    """Validate and filter chunks"""
    
    @staticmethod
    def validate_chunk(chunk: Dict[str, Any], config: ChunkConfig) -> bool:
        """Validate if chunk meets quality criteria"""
        text = chunk.get("text", "")
        
        # Check minimum size
        if len(text.split()) < config.min_chunk_size:
            return False
        
        # Check maximum size
        if len(text.split()) > config.max_chunk_size:
            return False
        
        # Use preprocessor to check meaningfulness
        preprocessor = TextPreprocessor()
        return preprocessor.is_meaningful_text(text)
    
    @staticmethod
    def remove_duplicate_chunks(chunks: List[Dict[str, Any]], similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Remove near-duplicate chunks based on text similarity"""
        if not chunks:
            return chunks
        
        def jaccard_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0
        
        filtered_chunks = []
        for chunk in chunks:
            is_duplicate = False
            for existing in filtered_chunks:
                if jaccard_similarity(chunk["text"], existing["text"]) > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_chunks.append(chunk)
        
        return filtered_chunks

# Enhanced chunking functions
def smart_fixed_chunk(text: str, config: ChunkConfig = None) -> List[Dict[str, Any]]:
    """Enhanced fixed chunking with sentence boundary awareness"""
    if config is None:
        config = ChunkConfig()
    
    preprocessor = TextPreprocessor()
    text = preprocessor.clean_text(text)
    
    if not preprocessor.is_meaningful_text(text):
        return []
    
    chunks = []
    
    if config.sentence_split:
        sentences = sent_tokenize(text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_size + sentence_size > config.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if preprocessor.is_meaningful_text(chunk_text):
                    chunks.append({
                        "id": _make_id(chunk_text),
                        "text": chunk_text,
                        "meta": {
                            "chunk_type": "smart_fixed",
                            "size": current_size,
                            "sentence_count": len(current_chunk)
                        }
                    })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-config.overlap//10:] if config.overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if preprocessor.is_meaningful_text(chunk_text):
                chunks.append({
                    "id": _make_id(chunk_text),
                    "text": chunk_text,
                    "meta": {
                        "chunk_type": "smart_fixed",
                        "size": current_size,
                        "sentence_count": len(current_chunk)
                    }
                })
    else:
        # Fallback to word-based chunking
        words = text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + config.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if preprocessor.is_meaningful_text(chunk_text):
                chunks.append({
                    "id": _make_id(chunk_text),
                    "text": chunk_text,
                    "meta": {"chunk_type": "smart_fixed", "size": len(chunk_words)}
                })
            i += config.chunk_size - config.overlap
    
    # Validate and filter chunks
    validator = ChunkValidator()
    chunks = [c for c in chunks if validator.validate_chunk(c, config)]
    
    if config.remove_duplicates:
        chunks = validator.remove_duplicate_chunks(chunks)
    
    return chunks

def hierarchical_chunk(text: str, levels: List[int] = None) -> List[Dict[str, Any]]:
    """Create chunks at multiple granularity levels"""
    if levels is None:
        levels = [128, 256, 512]  # Small, medium, large chunks
    
    preprocessor = TextPreprocessor()
    text = preprocessor.clean_text(text)
    
    all_chunks = []
    
    for level in levels:
        config = ChunkConfig(chunk_size=level, overlap=level//8)
        level_chunks = smart_fixed_chunk(text, config)
        
        for chunk in level_chunks:
            chunk["meta"]["granularity_level"] = level
            chunk["id"] = _make_id(chunk["text"], f"level_{level}")
            all_chunks.append(chunk)
    
    return all_chunks

def content_aware_chunk(text: str, content_type: str = "general") -> List[Dict[str, Any]]:
    """Chunk based on content type (FAQ, documentation, etc.)"""
    preprocessor = TextPreprocessor()
    text = preprocessor.clean_text(text)
    
    if content_type == "faq":
        # Split by Q&A patterns
        qa_pattern = r'(?:^|\n)(?:Q|Question|Query)[:.]?\s*(.*?)(?=(?:^|\n)(?:A|Answer|Response)[:.]?\s*)(.*?)(?=(?:^|\n)(?:Q|Question|Query)[:.]?|\Z)'
        matches = re.findall(qa_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        chunks = []
        for i, (question, answer) in enumerate(matches):
            qa_text = f"Q: {question.strip()}\nA: {answer.strip()}"
            if preprocessor.is_meaningful_text(qa_text):
                chunks.append({
                    "id": _make_id(qa_text, f"qa_{i}"),
                    "text": qa_text,
                    "meta": {
                        "chunk_type": "faq",
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "qa_pair": i + 1
                    }
                })
        
        if chunks:
            return chunks
    
    elif content_type == "list":
        # Split by list items
        list_items = re.split(r'\n(?=\s*[-•*]\s*|\s*\d+[\.)]\s*)', text)
        chunks = []
        
        for i, item in enumerate(list_items):
            item = item.strip()
            if preprocessor.is_meaningful_text(item):
                chunks.append({
                    "id": _make_id(item, f"list_{i}"),
                    "text": item,
                    "meta": {
                        "chunk_type": "list_item",
                        "item_number": i + 1
                    }
                })
        
        if chunks:
            return chunks
    
    # Fallback to smart fixed chunking
    return smart_fixed_chunk(text)

def adaptive_chunk(text: str, target_embedding_model: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """Adapt chunk size based on embedding model constraints"""
    # Model-specific optimal chunk sizes
    model_configs = {
        "all-MiniLM-L6-v2": ChunkConfig(chunk_size=384, overlap=50),
        "e5-small": ChunkConfig(chunk_size=450, overlap=60),
        "e5-large": ChunkConfig(chunk_size=500, overlap=70),
    }
    
    config = model_configs.get(target_embedding_model, ChunkConfig())
    return smart_fixed_chunk(text, config)

# Legacy function aliases for backward compatibility
def fixed_chunk(text, chunk_size=512, overlap=64):
    """Legacy wrapper for backward compatibility"""
    config = ChunkConfig(chunk_size=chunk_size, overlap=overlap)
    return smart_fixed_chunk(text, config)

def semantic_chunk(text, max_words=300, min_words=60):
    """Enhanced semantic chunking"""
    config = ChunkConfig(chunk_size=max_words, min_chunk_size=min_words)
    return smart_fixed_chunk(text, config)

def structural_chunk(html_text):
    """Enhanced structural chunking"""
    preprocessor = TextPreprocessor()
    
    # Try to detect headings and sections
    sections = re.split(r'\n(?=#{1,6}\s|\n(?=[A-Z][A-Z\s]+)\n)', html_text)
    
    chunks = []
    for i, section in enumerate(sections):
        section = preprocessor.clean_text(section)
        if preprocessor.is_meaningful_text(section):
            chunks.append({
                "id": _make_id(section, f"section_{i}"),
                "text": section,
                "meta": {"chunk_type": "structural", "section_number": i + 1}
            })
    
    if not chunks:
        # Fallback to paragraph splitting
        paras = html_text.split("\n\n")
        for i, p in enumerate(paras):
            p = preprocessor.clean_text(p)
            if preprocessor.is_meaningful_text(p):
                chunks.append({
                    "id": _make_id(p, f"para_{i}"),
                    "text": p,
                    "meta": {"chunk_type": "paragraph", "paragraph_number": i + 1}
                })
    
    return chunks

def recursive_chunk(text, threshold_words=600):
    """Enhanced recursive chunking"""
    struct_chunks = structural_chunk(text)
    out = []
    
    for chunk in struct_chunks:
        words = chunk["text"].split()
        if len(words) > threshold_words:
            # Break down large chunks further
            sub_chunks = smart_fixed_chunk(chunk["text"])
            for sub_chunk in sub_chunks:
                sub_chunk["meta"]["parent_chunk_id"] = chunk["id"]
                sub_chunk["meta"]["chunk_type"] = "recursive"
                out.append(sub_chunk)
        else:
            out.append(chunk)
    
    return out

def llm_chunk(text, model_func=None):
    """LLM-based chunking placeholder"""
    if model_func is None:
        return smart_fixed_chunk(text)
    return smart_fixed_chunk(text)