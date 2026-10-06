"""
OCEAN RAG
Retrieval Augmented Generation for Scientific Knowledge
OCEANVERSE Phase 5
"""

import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

class DocumentType(Enum):
    """Types of documents in knowledge base."""
    RESEARCH_PAPER = "research_paper"
    TECHNICAL_REPORT = "technical_report"
    DATASET_DESCRIPTION = "dataset_description"
    STANDARD_DEFINITION = "standard_definition"
    NEWS_ARTICLE = "news_article"
    TUTORIAL = "tutorial"
    BEST_PRACTICE = "best_practice"


@dataclass
class Document:
    """Scientific document."""
    id: str
    title: str
    authors: List[str]
    publication_date: datetime
    document_type: DocumentType
    text: str
    abstract: str
    keywords: List[str] = field(default_factory=list)
    doi: str = ""
    url: str = ""
    source: str = ""  # Copernicus, INCOIS, arXiv, etc.
    embeddings: Optional[np.ndarray] = None


@dataclass
class Citation:
    """Citation information."""
    doi: str = ""
    authors: List[str] = field(default_factory=list)
    title: str = ""
    year: int = 0
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    url: str = ""


@dataclass
class AnswerWithCitations:
    """Generated answer with supporting citations."""
    answer: str
    citations: List[Dict]  # {text, source, doi, authors, year}
    confidence: float
    sources_used: List[str]
    processing_time_ms: float


@dataclass
class TermDefinition:
    """Ocean science glossary term."""
    term: str
    formal_definition: str
    symbol: str = ""
    unit: str = ""
    examples: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# VECTOR STORE (Semantic Search Backend)
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """Semantic similarity search using embeddings."""
    
    def __init__(self, dimension: int = 1536):  # OpenAI embedding dimension
        self.dimension = dimension
        self.embeddings = np.empty((0, dimension), dtype=np.float32)
        self.document_ids = []
        self.metadata = {}
        self.logger = logger
    
    def add_document(self,
                    document_id: str,
                    text: str,
                    embedding: np.ndarray,
                    metadata: dict = None):
        """Add document embedding to vector store."""
        
        if len(embedding) != self.dimension:
            raise ValueError(f"Expected {self.dimension}D, got {len(embedding)}D")
        
        # Normalize embedding
        normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        self.embeddings = np.vstack([self.embeddings, normalized.reshape(1, -1)])
        self.document_ids.append(document_id)
        self.metadata[document_id] = metadata or {}
    
    def search(self,
              query_embedding: np.ndarray,
              top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        
        if len(self.embeddings) == 0:
            return []
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc_id = self.document_ids[idx]
            similarity = float(similarities[idx])
            results.append((doc_id, similarity))
        
        return results


# ─────────────────────────────────────────────────────────────
# DOCUMENT STORE
# ─────────────────────────────────────────────────────────────

class DocumentStore:
    """Manage scientific documents and metadata."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.logger = logger
        self.glossary = {}
    
    def add_document(self,
                    source_url: str,
                    title: str,
                    authors: List[str],
                    publication_date: datetime,
                    text: str,
                    document_type: DocumentType,
                    abstract: str = "",
                    keywords: List[str] = None,
                    doi: str = "") -> str:
        """Index a new document."""
        
        doc_id = f"doc_{len(self.documents):06d}"
        
        document = Document(
            id=doc_id,
            title=title,
            authors=authors,
            publication_date=publication_date,
            document_type=document_type,
            text=text,
            abstract=abstract,
            keywords=keywords or [],
            doi=doi,
            url=source_url,
        )
        
        self.documents[doc_id] = document
        self.logger.info(f"Added document: {doc_id} - {title}")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def search_keywords(self,
                       keywords: List[str],
                       doc_type: DocumentType = None) -> List[str]:
        """Search documents by keywords."""
        
        results = []
        for doc_id, doc in self.documents.items():
            if doc_type and doc.document_type != doc_type:
                continue
            
            if any(kw.lower() in [k.lower() for k in doc.keywords] for kw in keywords):
                results.append(doc_id)
        
        return results
    
    def add_glossary_term(self,
                         term: str,
                         definition: str,
                         symbol: str = "",
                         unit: str = "",
                         examples: List[str] = None,
                         related_terms: List[str] = None):
        """Add oceanographic term to glossary."""
        
        self.glossary[term.lower()] = TermDefinition(
            term=term,
            formal_definition=definition,
            symbol=symbol,
            unit=unit,
            examples=examples or [],
            related_terms=related_terms or [],
        )
    
    def get_glossary_term(self, term: str) -> Optional[TermDefinition]:
        """Retrieve glossary definition."""
        return self.glossary.get(term.lower())
    
    def list_documents(self,
                      filters: dict = None) -> List[Document]:
        """Filter documents by metadata."""
        
        results = []
        for doc in self.documents.values():
            if filters is None:
                results.append(doc)
                continue
            
            match = True
            if 'type' in filters and doc.document_type != filters['type']:
                match = False
            if 'author' in filters and filters['author'] not in doc.authors:
                match = False
            if 'year' in filters and doc.publication_date.year != filters['year']:
                match = False
            
            if match:
                results.append(doc)
        
        return results


# ─────────────────────────────────────────────────────────────
# CITATION MANAGER
# ─────────────────────────────────────────────────────────────

class CitationManager:
    """Manage citations and bibliography."""
    
    @staticmethod
    def format_citation(doc: Document, style: str = 'apa') -> str:
        """Generate citation in requested format."""
        
        authors = ", ".join(doc.authors[:3])
        if len(doc.authors) > 3:
            authors += " et al."
        
        if style == 'apa':
            return (
                f"{authors} ({doc.publication_date.year}). {doc.title}. "
                f"Retrieved from {doc.url}"
            )
        elif style == 'chicago':
            return (
                f"{authors}. {doc.publication_date.year}. \"{doc.title}.\" "
                f"Accessed from {doc.url}."
            )
        elif style == 'mla':
            return (
                f"{authors}. \"{doc.title}.\" {doc.publication_date.year}. "
                f"Web. {doc.url}"
            )
        else:
            return f"{authors} ({doc.publication_date.year}): {doc.title}"
    
    @staticmethod
    def extract_doi_metadata(doi: str) -> Dict:
        """Fetch metadata from DOI resolver."""
        # Simplified: in production, call crossref API
        return {
            'doi': doi,
            'resolved': True,
            'metadata': {},
        }


# ─────────────────────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────────────────────

class OceanRAG:
    """Retrieval Augmented Generation for ocean science."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.document_store = DocumentStore()
        self.citation_manager = CitationManager()
        self.logger = logger
    
    def semantic_search(self,
                       query: str,
                       top_k: int = 5,
                       min_relevance: float = 0.5) -> List[Dict]:
        """
        Search documents by semantic meaning.
        
        Returns:
            List of {doc_id, title, relevance_score, excerpt}
        """
        
        # In production: use OpenAI embeddings API
        # For now: simulate with keyword matching
        query_lower = query.lower()
        query_words = query_lower.split()
        
        results = []
        for doc_id, doc in self.document_store.documents.items():
            # Score based on keyword matches
            score = sum(
                1.0 for word in query_words
                if word in doc.title.lower() or word in doc.text.lower()
            ) / len(query_words)
            
            if score >= min_relevance:
                excerpt = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                results.append({
                    'doc_id': doc_id,
                    'title': doc.title,
                    'relevance': score,
                    'excerpt': excerpt,
                    'authors': doc.authors,
                    'year': doc.publication_date.year,
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:top_k]
    
    def generate_answer(self,
                       question: str,
                       context_docs: List[str] = None,
                       use_lm: bool = False) -> AnswerWithCitations:
        """
        Generate answer using RAG.
        
        Process:
        1. Retrieve relevant documents
        2. Format context
        3. Generate answer with LLM
        4. Extract citations
        """
        
        start_time = datetime.now()
        
        # Retrieve documents
        if context_docs is None:
            search_results = self.semantic_search(question, top_k=3)
            context_docs = [r['doc_id'] for r in search_results]
        
        # Prepare context
        context_text = ""
        citations = []
        
        for doc_id in context_docs:
            doc = self.document_store.get_document(doc_id)
            if doc:
                context_text += f"\n\n[{doc.title}]\n{doc.text[:500]}"
                citations.append({
                    'title': doc.title,
                    'authors': doc.authors,
                    'year': doc.publication_date.year,
                    'doi': doc.doi,
                    'url': doc.url,
                })
        
        # Generate answer (simplified)
        answer = self._generate_answer_text(question, context_text)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AnswerWithCitations(
            answer=answer,
            citations=citations,
            confidence=0.85,
            sources_used=context_docs,
            processing_time_ms=processing_time,
        )
    
    def _generate_answer_text(self, question: str, context: str) -> str:
        """
        Generate answer text (simplified version).
        In production: call Claude/GPT with context.
        """
        
        # Placeholder: simple retrieval
        if context:
            return (
                f"Based on the scientific literature, {question.lower()} "
                f"The available research suggests: {context[:300]}..."
            )
        return f"I could not find sufficient information to answer: {question}"
    
    def explain_prediction(self,
                          prediction: Dict,
                          variable: str = None) -> AnswerWithCitations:
        """Explain why model made this prediction."""
        
        question = f"Why is {variable} at {prediction.get('value')} in these conditions?"
        return self.generate_answer(question)
    
    def retrieve_similar_cases(self,
                              current_state: Dict,
                              k: int = 3) -> List[Dict]:
        """Find similar past ocean states."""
        
        # Query: find documents about similar conditions
        query_parts = []
        if 'temperature' in current_state:
            query_parts.append(f"temperature {current_state['temperature']}")
        if 'event_type' in current_state:
            query_parts.append(current_state['event_type'])
        
        query = " ".join(query_parts)
        return self.semantic_search(query, top_k=k)
    
    def get_glossary_term(self, term: str) -> Optional[TermDefinition]:
        """Get standardized definition."""
        return self.document_store.get_glossary_term(term)


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE BASE INITIALIZATION
# ─────────────────────────────────────────────────────────────

class KnowledgeBaseBuilder:
    """Build ocean science knowledge base."""
    
    def __init__(self, rag: OceanRAG):
        self.rag = rag
        self.logger = logger
    
    def bootstrap_glossary(self):
        """Initialize ocean science glossary."""
        
        terms = [
            {
                'term': 'Sea Surface Temperature (SST)',
                'definition': 'Temperature of ocean water at the surface, measured by satellites or buoys',
                'symbol': 'SST',
                'unit': '°C or K',
                'examples': ['Typical SST in Indian Ocean: 25-30°C'],
                'related': ['Anomaly', 'Heat waves'],
            },
            {
                'term': 'Monsoon',
                'definition': 'Seasonal reversing wind pattern that dramatically influences ocean currents and rainfall',
                'symbol': 'SW/NE Monsoon',
                'examples': ['Indian Summer Monsoon (June-September)'],
                'related': ['Upwelling', 'Currents'],
            },
            {
                'term': 'Thermocline',
                'definition': 'Layer of water with rapid temperature decrease with depth',
                'symbol': 'T(z)',
                'unit': 'degrees/meter',
                'related': ['Stratification', 'Density'],
            },
            {
                'term': 'Salinity',
                'definition': 'Measure of salt concentration in seawater',
                'symbol': 'S',
                'unit': 'PSU (Practical Salinity Units)',
                'examples': ['Typical ocean salinity: 35 PSU'],
                'related': ['Density', 'Stratification'],
            },
        ]
        
        for term_data in terms:
            self.rag.document_store.add_glossary_term(
                term=term_data['term'],
                definition=term_data['definition'],
                symbol=term_data.get('symbol', ''),
                unit=term_data.get('unit', ''),
                examples=term_data.get('examples', []),
                related_terms=term_data.get('related', []),
            )
        
        self.logger.info(f"Initialized glossary with {len(terms)} terms")
    
    def add_copernicus_documents(self):
        """Index documents from Copernicus."""
        # Placeholder for data ingestion
        pass
    
    def add_incois_documents(self):
        """Index Indian National Centre for Ocean Information Services documents."""
        # Placeholder for data ingestion
        pass
    
    def add_research_papers(self):
        """Index research papers from arXiv/journals."""
        # Placeholder for data ingestion
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    rag = OceanRAG()
    builder = KnowledgeBaseBuilder(rag)
    builder.bootstrap_glossary()
