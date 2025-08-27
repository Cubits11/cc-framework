# src/cc/guardrails/semantic_filter.py
"""Semantic similarity based guardrail"""
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import Guardrail

class SemanticFilter(Guardrail):
    """Semantic similarity based content filter"""
    
    def __init__(self, 
                 harmful_templates: List[str],
                 similarity_threshold: float = 0.8,
                 vectorizer_params: Optional[dict] = None):
        
        self.harmful_templates = harmful_templates
        self.similarity_threshold = similarity_threshold
        
        # Initialize TF-IDF vectorizer
        default_params = {
            'max_features': 5000,
            'ngram_range': (1, 3),
            'stop_words': 'english'
        }
        if vectorizer_params:
            default_params.update(vectorizer_params)
        
        self.vectorizer = TfidfVectorizer(**default_params)
        
        # Fit on harmful templates
        if harmful_templates:
            self.template_vectors = self.vectorizer.fit_transform(harmful_templates)
        else:
            self.template_vectors = None
    
    def blocks(self, text: str) -> bool:
        """Check if text is semantically similar to harmful templates"""
        return self.score(text) > self.similarity_threshold
    
    def score(self, text: str) -> float:
        """Compute maximum cosine similarity to harmful templates"""
        if not self.template_vectors or not text.strip():
            return 0.0
        
        try:
            # Vectorize input text
            text_vector = self.vectorizer.transform([text])
            
            # Compute similarities
            similarities = cosine_similarity(text_vector, self.template_vectors)
            
            # Return maximum similarity
            return float(np.max(similarities))
            
        except Exception:
            # Fallback for unseen vocabulary
            return 0.0
    
    def calibrate(self, benign_texts: list, target_fpr: float = 0.05) -> None:
        """Calibrate similarity threshold to achieve target FPR"""
        if not benign_texts or not self.template_vectors:
            return
        
        scores = [self.score(text) for text in benign_texts]
        scores.sort(reverse=True)
        
        # Find threshold
        n = len(scores)
        target_idx = int(target_fpr * n)
        
        if target_idx < n:
            self.similarity_threshold = max(0.1, scores[target_idx])  # Minimum threshold
        else:
            self.similarity_threshold = 0.99  # Very high threshold
        
        # Validate
        actual_fpr = sum(1 for s in scores if s > self.similarity_threshold) / n
        print(f"SemanticFilter calibrated: threshold={self.similarity_threshold:.3f}, FPR={actual_fpr:.3f}")