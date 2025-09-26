# backend/generator.py
"""
Enhanced answer generator supporting multiple LLMs for grounded question answering.
Supports local models (Flan-T5, T5) and API-based models (OpenAI, Anthropic).
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Optional imports
try:
    import openai
    _openai_available = True
except ImportError:
    openai = None
    _openai_available = False

try:
    from anthropic import Anthropic
    _anthropic_available = True
except ImportError:
    Anthropic = None
    _anthropic_available = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 1
    early_stopping: bool = True
    do_sample: bool = True
    repetition_penalty: float = 1.05


class BaseGenerator(ABC):
    """Abstract base class for answer generators."""
    
    @abstractmethod
    def generate_answer(self, 
                       query: str, 
                       retrieved: List[Dict[str, Any]],
                       config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class HuggingFaceGenerator(BaseGenerator):
    """Generator using Hugging Face transformers models."""

    MODELS = {
        'flan-t5-small': {'name': 'google/flan-t5-small', 'type': 'seq2seq', 'description': 'Small, fast, good for testing', 'context_length': 512},
        'flan-t5-base': {'name': 'google/flan-t5-base', 'type': 'seq2seq', 'description': 'Better quality than small', 'context_length': 512},
        'flan-t5-large': {'name': 'google/flan-t5-large', 'type': 'seq2seq', 'description': 'High quality, slower', 'context_length': 512},
        't5-small': {'name': 't5-small', 'type': 'seq2seq', 'description': 'Basic T5 model', 'context_length': 512}
    }
    
    def __init__(self, model_name: str = 'flan-t5-base', device: Optional[str] = None, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        logger.info(f"Initialized HuggingFaceGenerator: {model_name} on {device}")

    def load_model(self) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        if self.is_loaded:
            return self.tokenizer, self.model
        if self.model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(self.MODELS.keys())}")
        model_path = self.MODELS[self.model_name]['name']
        try:
            logger.info(f"Loading {model_path}...")
            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
            self.is_loaded = True
            return self.tokenizer, self.model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise

    def build_prompt(self, query: str, retrieved: List[Dict[str, Any]], max_context_chars: int = 2000, include_urls: bool = True, include_confidence: bool = True) -> str:
        if not retrieved:
            return (f"You are a helpful customer support assistant for JioPay. User question: {query}\n\n"
                    "No relevant information found. Please say: 'I don't have information about this. Please check JioPay official support.'\n\nAnswer:")
        context_pieces, char_count = [], 0
        for i, chunk in enumerate(retrieved):
            meta = chunk.get('meta', {})
            text = meta.get('text', '')
            snippet = text[:600] if len(text) > 600 else text
            source_info = []
            if include_urls and meta.get('url'):
                source_info.append(f"URL: {meta['url']}")
            if meta.get('title'):
                source_info.append(f"Title: {meta['title']}")
            if include_confidence and 'score' in chunk:
                source_info.append(f"Relevance: {chunk['score']:.3f}")
            piece = f"[Source {i+1}]"
            if source_info:
                piece += f" ({', '.join(source_info)})"
            piece += f"\n{snippet}"
            if char_count + len(piece) > max_context_chars:
                break
            context_pieces.append(piece)
            char_count += len(piece)
        context = "\n\n---\n\n".join(context_pieces)
        return (f"You are a helpful customer support assistant for JioPay. "
        f"Answer the user's question using ONLY the information provided in the sources below. "
        f"Provide a detailed, well-structured explanation. "
        f"Break down complex ideas into simple steps and include examples if possible.\n\n"
        f"User Question: {query}\n\nSources:\n{context}\n\nDetailed Answer:")


    def generate_answer(self, query: str, retrieved: List[Dict[str, Any]], config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        if config is None: config = GenerationConfig()
        if not self.is_loaded: self.load_model()
        prompt = self.build_prompt(query, retrieved)
        start_time = time.time()
        try:
            if self.pipeline:
                outputs = self.pipeline(prompt, max_new_tokens=config.max_new_tokens, temperature=config.temperature, top_p=config.top_p,
                                        num_beams=config.num_beams, early_stopping=config.early_stopping, do_sample=config.do_sample,
                                        repetition_penalty=config.repetition_penalty)
                answer = outputs[0]['generated_text']
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=config.max_new_tokens, temperature=config.temperature,
                                                  top_p=config.top_p, num_beams=config.num_beams, early_stopping=config.early_stopping,
                                                  do_sample=config.do_sample, repetition_penalty=config.repetition_penalty)
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_time = time.time() - start_time
            citations = [{'index': i, 'url': c.get('meta', {}).get('url', ''), 'title': c.get('meta', {}).get('title', ''), 'snippet': c.get('meta', {}).get('text', '')[:200], 'score': c.get('score', 0.0)} for i, c in enumerate(retrieved)]
            return {'answer': answer, 'citations': citations, 'generation_time': generation_time, 'model': self.model_name, 'prompt_length': len(prompt), 'context_sources': len(retrieved)}
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {'answer': "I encountered an error while generating the answer. Please try again or contact support.", 'citations': [], 'generation_time': 0, 'model': self.model_name, 'error': str(e)}

    def is_available(self) -> bool:
        return self.is_loaded and self.model is not None


# Only define OpenAIGenerator if available
if _openai_available:
    class OpenAIGenerator(BaseGenerator):
        MODELS = {'gpt-3.5-turbo': {'context_length': 4096, 'cost_per_1k': 0.002}, 'gpt-4': {'context_length': 8192, 'cost_per_1k': 0.03}, 'gpt-4-turbo': {'context_length': 128000, 'cost_per_1k': 0.01}}
        def __init__(self, model_name: str = 'gpt-3.5-turbo', api_key: Optional[str] = None):
            self.model_name = model_name
            self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        def generate_answer(self, query: str, retrieved: List[Dict[str, Any]], config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
            if config is None: config = GenerationConfig()
            context = self._build_context(retrieved)
            messages = [{"role": "system", "content": "You are a helpful assistant for JioPay. Answer questions using only the provided context."},
                        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"}]
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(model=self.model_name, messages=messages, max_tokens=config.max_new_tokens, temperature=config.temperature, top_p=config.top_p)
                generation_time = time.time() - start_time
                answer = response.choices[0].message.content
                return {'answer': answer, 'citations': self._extract_citations(retrieved), 'generation_time': generation_time, 'model': self.model_name, 'tokens_used': response.usage.total_tokens, 'cost_estimate': response.usage.total_tokens * self.MODELS[self.model_name]['cost_per_1k'] / 1000}
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return {'answer': "I encountered an error. Please try again.", 'citations': [], 'error': str(e)}
        def _build_context(self, retrieved: List[Dict]) -> str:
            return "\n\n".join([f"[{i+1}] {c.get('meta', {}).get('text', '')[:500]} (Source: {c.get('meta', {}).get('url', '')})" for i, c in enumerate(retrieved)])
        def _extract_citations(self, retrieved: List[Dict]) -> List[Dict]:
            return [{'index': i, 'url': c.get('meta', {}).get('url', ''), 'title': c.get('meta', {}).get('title', ''), 'snippet': c.get('meta', {}).get('text', '')[:200], 'score': c.get('score', 0.0)} for i, c in enumerate(retrieved)]
        def is_available(self) -> bool:
            return bool(os.getenv('OPENAI_API_KEY'))
else:
    class OpenAIGenerator(BaseGenerator):
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenAI not installed. Run `pip install openai` to enable this generator.")
        def generate_answer(self, *args, **kwargs): return {"answer": "OpenAI not available."}
        def is_available(self) -> bool: return False


def create_generator(provider: str = 'huggingface', model_name: str = 'flan-t5-base', **kwargs) -> BaseGenerator:
    if provider == 'huggingface':
        return HuggingFaceGenerator(model_name=model_name, **kwargs)
    elif provider == 'openai':
        if not _openai_available:
            raise ImportError("OpenAI package not installed. Run `pip install openai` to enable it.")
        return OpenAIGenerator(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Backward compatible functions
_default_generator = None
def load_model():
    global _default_generator
    if _default_generator is None:
        _default_generator = HuggingFaceGenerator(model_name="flan-t5-base")
        _default_generator.load_model()
    return _default_generator.tokenizer, _default_generator.model

def build_prompt(query: str, retrieved: List[Dict], max_context_chars: int = 3000) -> str:
    generator = HuggingFaceGenerator(model_name="flan-t5-base")
    return generator.build_prompt(query, retrieved, max_context_chars)

def generate_answer(query: str, retrieved: List[Dict], max_new_tokens: int = 200) -> str:
    global _default_generator
    if _default_generator is None:
        _default_generator = HuggingFaceGenerator(model_name="flan-t5-base")
    config = GenerationConfig(max_new_tokens=max_new_tokens)
    result = _default_generator.generate_answer(query, retrieved, config)
    return result['answer']
