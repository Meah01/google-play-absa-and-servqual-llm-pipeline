"""
Deep ABSA Analysis Engine for ABSA Pipeline.
Implements comprehensive aspect-based sentiment analysis using RoBERTa and spaCy.
Optimized for ecommerce app review analysis with high accuracy for business intelligence.
Integrates with SERVQUAL service quality framework.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
import spacy
from spacy.tokens import Doc, Span, Token

from src.utils.config import config
from src.absa.models import get_deep_absa_models, model_manager
from src.absa.preprocessing import ABSAPreprocessor, PreprocessingResult


@dataclass
class AspectResult:
    """Result of aspect-level sentiment analysis."""
    aspect: str
    sentiment_score: float  # -1.0 to +1.0
    confidence_score: float  # 0.0 to 1.0
    opinion_text: str
    opinion_start_pos: int
    opinion_end_pos: int
    contributing_keywords: List[str]


@dataclass
class ReviewAnalysisResult:
    """Complete ABSA analysis result for a single review."""
    review_id: str
    app_id: str
    original_text: str
    processed_text: str
    aspects: List[AspectResult]
    processing_time_ms: int
    processing_model: str
    processing_version: str
    analysis_metadata: Dict[str, Any]


class AspectExtractor:
    """Extracts aspects and their associated opinion text from reviews."""

    def __init__(self, spacy_model):
        self.logger = logging.getLogger("absa_pipeline.deep_engine.aspects")
        self.nlp = spacy_model
        self.aspect_keywords = {}
        self.aspect_patterns = {}
        self._load_aspect_definitions()

    def _load_aspect_definitions(self):
        """Load aspect keywords and patterns from database."""
        try:
            from src.data.storage import storage

            query = """
            SELECT aspect_name, keywords, servqual_dimension, weight, description
            FROM aspects 
            WHERE is_active = TRUE
            ORDER BY weight DESC
            """

            df = storage.db.execute_query(query)

            for _, row in df.iterrows():
                aspect_name = row['aspect_name']
                keywords = row['keywords'] if row['keywords'] else []

                # Store keywords for this aspect
                self.aspect_keywords[aspect_name] = [kw.lower() for kw in keywords]

                # Create regex patterns for better matching
                keyword_patterns = []
                for keyword in keywords:
                    # Create case-insensitive pattern with word boundaries
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    keyword_patterns.append(pattern)

                if keyword_patterns:
                    self.aspect_patterns[aspect_name] = '|'.join(keyword_patterns)

            self.logger.info(f"Loaded {len(self.aspect_keywords)} aspect definitions")

        except Exception as e:
            self.logger.error(f"Error loading aspect definitions: {e}")
            # Fallback to basic aspects
            self.aspect_keywords = {
                'product_quality': ['quality', 'cheap', 'durable', 'defective'],
                'shipping_delivery': ['shipping', 'delivery', 'fast', 'slow'],
                'customer_service': ['service', 'support', 'help']
            }
            self.aspect_patterns = {}

    def extract_aspects_with_context(self, text: str, doc: Doc = None) -> List[Tuple[str, str, int, int]]:
        """
        Extract aspects with their contextual opinion text and positions.

        Args:
            text: Preprocessed review text
            doc: Optional spaCy Doc object

        Returns:
            List of tuples: (aspect_name, opinion_text, start_pos, end_pos)
        """
        if doc is None:
            doc = self.nlp(text)

        aspect_extractions = []
        text_lower = text.lower()

        # Extract aspects using keyword matching with context
        for aspect_name, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()

                # Find all occurrences of this keyword
                start_idx = 0
                while True:
                    pos = text_lower.find(keyword_lower, start_idx)
                    if pos == -1:
                        break

                    # Extract context around the keyword
                    opinion_text, opinion_start, opinion_end = self._extract_opinion_context(
                        text, doc, pos, len(keyword_lower)
                    )

                    if opinion_text.strip():
                        aspect_extractions.append((
                            aspect_name,
                            opinion_text.strip(),
                            opinion_start,
                            opinion_end
                        ))

                    start_idx = pos + 1

        # Deduplicate overlapping extractions (prefer longer opinions)
        deduplicated = self._deduplicate_extractions(aspect_extractions)

        return deduplicated

    def _extract_opinion_context(self, text: str, doc: Doc, keyword_pos: int, keyword_len: int) -> Tuple[str, int, int]:
        """Extract opinion context around a keyword."""
        try:
            # Find the sentence containing the keyword
            keyword_end = keyword_pos + keyword_len

            # Find sentence boundaries using spaCy
            target_sent = None
            for sent in doc.sents:
                if sent.start_char <= keyword_pos < sent.end_char:
                    target_sent = sent
                    break

            if target_sent is None:
                # Fallback: extract fixed window around keyword
                window_size = 50
                start_pos = max(0, keyword_pos - window_size)
                end_pos = min(len(text), keyword_end + window_size)
                return text[start_pos:end_pos], start_pos, end_pos

            # Extract the full sentence as opinion text
            opinion_text = target_sent.text
            opinion_start = target_sent.start_char
            opinion_end = target_sent.end_char

            return opinion_text, opinion_start, opinion_end

        except Exception as e:
            self.logger.debug(f"Error extracting opinion context: {e}")
            # Fallback to simple window extraction
            window_size = 30
            start_pos = max(0, keyword_pos - window_size)
            end_pos = min(len(text), keyword_end + window_size)
            return text[start_pos:end_pos], start_pos, end_pos

    def _deduplicate_extractions(self, extractions: List[Tuple[str, str, int, int]]) -> List[Tuple[str, str, int, int]]:
        """Remove overlapping extractions, preferring longer opinion text."""
        if not extractions:
            return []

        # Sort by opinion length (longer first)
        sorted_extractions = sorted(extractions, key=lambda x: len(x[1]), reverse=True)

        deduplicated = []
        used_ranges = []

        for aspect, opinion, start, end in sorted_extractions:
            # Check if this extraction overlaps significantly with existing ones
            overlaps = False
            for used_start, used_end in used_ranges:
                overlap_start = max(start, used_start)
                overlap_end = min(end, used_end)
                overlap_length = max(0, overlap_end - overlap_start)

                # If more than 50% overlap, consider it duplicate
                current_length = end - start
                if overlap_length > 0.5 * current_length:
                    overlaps = True
                    break

            if not overlaps:
                deduplicated.append((aspect, opinion, start, end))
                used_ranges.append((start, end))

        return deduplicated


class SentimentAnalyzer:
    """Analyzes sentiment for aspect-opinion pairs using RoBERTa."""

    def __init__(self, sentiment_model, tokenizer):
        self.logger = logging.getLogger("absa_pipeline.deep_engine.sentiment")
        self.model = sentiment_model
        self.tokenizer = tokenizer
        self.sentiment_pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the sentiment analysis pipeline."""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True,
                truncation=True,
                max_length=config.absa.max_text_length
            )
            self.logger.debug("Sentiment analysis pipeline initialized")

        except Exception as e:
            self.logger.error(f"Error initializing sentiment pipeline: {e}")
            raise

    def analyze_aspect_sentiment(self, opinion_text: str, aspect: str) -> Tuple[float, float]:
        """
        Analyze sentiment for an aspect-opinion pair.

        Args:
            opinion_text: The opinion text mentioning the aspect
            aspect: The aspect name for context

        Returns:
            Tuple of (sentiment_score, confidence_score)
        """
        try:
            if not opinion_text or not opinion_text.strip():
                return 0.0, 0.0

            # Enhance the text with aspect context for better sentiment analysis
            enhanced_text = self._enhance_text_with_aspect(opinion_text, aspect)

            # Get sentiment predictions
            results = self.sentiment_pipeline(enhanced_text)

            # Convert to standardized sentiment score (-1 to +1)
            sentiment_score, confidence = self._convert_to_sentiment_score(results[0])

            return sentiment_score, confidence

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for aspect '{aspect}': {e}")
            return 0.0, 0.0

    def _enhance_text_with_aspect(self, opinion_text: str, aspect: str) -> str:
        """Enhance opinion text with aspect context for better sentiment analysis."""
        # For RoBERTa models, we can provide aspect context
        aspect_display = aspect.replace('_', ' ')
        enhanced = f"Regarding {aspect_display}: {opinion_text}"

        # Ensure we don't exceed max length
        if len(enhanced) > config.absa.max_text_length:
            # Truncate the opinion text to fit
            max_opinion_len = config.absa.max_text_length - len(f"Regarding {aspect_display}: ")
            truncated_opinion = opinion_text[:max_opinion_len]
            enhanced = f"Regarding {aspect_display}: {truncated_opinion}"

        return enhanced

    def _convert_to_sentiment_score(self, predictions: List[Dict]) -> Tuple[float, float]:
        """Convert model predictions to standardized sentiment score."""
        try:
            # Handle different model output formats
            label_mapping = {
                'POSITIVE': 1.0,
                'NEGATIVE': -1.0,
                'NEUTRAL': 0.0,
                'LABEL_0': -1.0,  # Negative
                'LABEL_1': 0.0,   # Neutral
                'LABEL_2': 1.0,   # Positive
                'NEG': -1.0,
                'POS': 1.0
            }

            # Calculate weighted sentiment score
            total_score = 0.0
            max_confidence = 0.0

            for pred in predictions:
                label = pred['label']
                confidence = pred['score']

                if label in label_mapping:
                    sentiment_value = label_mapping[label]
                    total_score += sentiment_value * confidence
                    max_confidence = max(max_confidence, confidence)

            # Normalize to -1 to +1 range
            sentiment_score = np.clip(total_score, -1.0, 1.0)

            return float(sentiment_score), float(max_confidence)

        except Exception as e:
            self.logger.error(f"Error converting sentiment predictions: {e}")
            return 0.0, 0.0

    def analyze_batch_sentiments(self, opinion_texts: List[str], aspects: List[str]) -> List[Tuple[float, float]]:
        """Analyze sentiment for a batch of opinion-aspect pairs."""
        results = []

        # Process in batches to manage memory
        batch_size = min(16, len(opinion_texts))  # Smaller batches for memory efficiency

        for i in range(0, len(opinion_texts), batch_size):
            batch_opinions = opinion_texts[i:i+batch_size]
            batch_aspects = aspects[i:i+batch_size]

            batch_results = []
            for opinion, aspect in zip(batch_opinions, batch_aspects):
                sentiment, confidence = self.analyze_aspect_sentiment(opinion, aspect)
                batch_results.append((sentiment, confidence))

            results.extend(batch_results)

        return results


class DeepABSAEngine:
    """Main deep ABSA analysis engine combining aspect extraction and sentiment analysis."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.deep_engine")

        # Initialize components
        self.preprocessor = ABSAPreprocessor()
        self.models_loaded = False
        self.sentiment_model = None
        self.tokenizer = None
        self.spacy_model = None
        self.aspect_extractor = None
        self.sentiment_analyzer = None

        # Configuration
        self.batch_size = config.absa.batch_size
        self.confidence_threshold = config.absa.confidence_threshold
        self.processing_model = config.absa.deep_model_name
        self.processing_version = "1.0.0"

        self.logger.info("Deep ABSA Engine initialized")

    def _load_models(self):
        """Load all required models for deep ABSA analysis."""
        if self.models_loaded:
            return

        try:
            self.logger.info("Loading deep ABSA models")

            # Load models using model manager
            self.sentiment_model, self.tokenizer, self.spacy_model = get_deep_absa_models()

            # Initialize components
            self.aspect_extractor = AspectExtractor(self.spacy_model)
            self.sentiment_analyzer = SentimentAnalyzer(self.sentiment_model, self.tokenizer)

            self.models_loaded = True
            self.logger.info("Deep ABSA models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading deep ABSA models: {e}")
            raise

    def analyze_single_review(self, review_id: str, app_id: str, review_text: str) -> ReviewAnalysisResult:
        """
        Analyze a single review for aspects and sentiment.

        Args:
            review_id: Unique review identifier
            app_id: Application identifier
            review_text: Review content to analyze

        Returns:
            Complete analysis result
        """
        start_time = time.time()

        try:
            # Ensure models are loaded
            self._load_models()

            # Preprocess the review text
            preprocessing_result = self.preprocessor.preprocess_single_review(review_text)

            if preprocessing_result.preprocessing_flags['is_empty_after_cleaning']:
                return self._create_empty_result(review_id, app_id, review_text, start_time)

            # Use the deep-processed text for analysis
            processed_text = preprocessing_result.deep_processed_text

            # Create spaCy document for linguistic analysis
            doc = self.spacy_model(processed_text)

            # Extract aspects with their opinion contexts
            aspect_extractions = self.aspect_extractor.extract_aspects_with_context(processed_text, doc)

            if not aspect_extractions:
                return self._create_empty_result(review_id, app_id, review_text, start_time)

            # Analyze sentiment for each aspect-opinion pair
            aspect_results = []

            for aspect_name, opinion_text, start_pos, end_pos in aspect_extractions:
                sentiment_score, confidence_score = self.sentiment_analyzer.analyze_aspect_sentiment(
                    opinion_text, aspect_name
                )

                # Only include results above confidence threshold
                if confidence_score >= self.confidence_threshold:
                    # Extract contributing keywords
                    keywords = self._extract_contributing_keywords(opinion_text, aspect_name)

                    aspect_result = AspectResult(
                        aspect=aspect_name,
                        sentiment_score=sentiment_score,
                        confidence_score=confidence_score,
                        opinion_text=opinion_text,
                        opinion_start_pos=start_pos,
                        opinion_end_pos=end_pos,
                        contributing_keywords=keywords
                    )

                    aspect_results.append(aspect_result)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Create analysis metadata
            metadata = {
                'preprocessing_stats': preprocessing_result.text_stats,
                'preprocessing_flags': preprocessing_result.preprocessing_flags,
                'detected_aspects_count': len(preprocessing_result.detected_aspects),
                'extracted_aspects_count': len(aspect_extractions),
                'final_aspects_count': len(aspect_results),
                'spacy_model': str(self.spacy_model.meta.get('name', 'unknown')),
                'confidence_threshold': self.confidence_threshold
            }

            return ReviewAnalysisResult(
                review_id=review_id,
                app_id=app_id,
                original_text=review_text,
                processed_text=processed_text,
                aspects=aspect_results,
                processing_time_ms=processing_time_ms,
                processing_model=self.processing_model,
                processing_version=self.processing_version,
                analysis_metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error analyzing review {review_id}: {e}")
            return self._create_error_result(review_id, app_id, review_text, start_time, str(e))

    def analyze_batch(self, reviews: List[Dict[str, str]]) -> List[ReviewAnalysisResult]:
        """
        Analyze a batch of reviews efficiently.

        Args:
            reviews: List of dicts with keys: review_id, app_id, content

        Returns:
            List of analysis results
        """
        self.logger.info(f"Starting batch analysis of {len(reviews)} reviews")
        start_time = time.time()

        # Ensure models are loaded
        self._load_models()

        results = []
        processed_count = 0
        error_count = 0

        for review in reviews:
            try:
                result = self.analyze_single_review(
                    review['review_id'],
                    review['app_id'],
                    review['content']
                )
                results.append(result)
                processed_count += 1

                if processed_count % 10 == 0:
                    self.logger.debug(f"Processed {processed_count}/{len(reviews)} reviews")

            except Exception as e:
                self.logger.error(f"Error processing review {review.get('review_id', 'unknown')}: {e}")
                error_count += 1

                # Create error result to maintain batch integrity
                error_result = self._create_error_result(
                    review.get('review_id', 'unknown'),
                    review.get('app_id', 'unknown'),
                    review.get('content', ''),
                    time.time(),
                    str(e)
                )
                results.append(error_result)

        total_time = time.time() - start_time

        self.logger.info(f"Batch analysis complete: {processed_count} processed, "
                        f"{error_count} errors, {total_time:.2f}s total")

        return results

    def _extract_contributing_keywords(self, opinion_text: str, aspect_name: str) -> List[str]:
        """Extract keywords that contributed to aspect detection."""
        keywords = []
        opinion_lower = opinion_text.lower()

        aspect_keywords = self.aspect_extractor.aspect_keywords.get(aspect_name, [])

        for keyword in aspect_keywords:
            if keyword.lower() in opinion_lower:
                keywords.append(keyword)

        return keywords

    def _create_empty_result(self, review_id: str, app_id: str, review_text: str, start_time: float) -> ReviewAnalysisResult:
        """Create empty result for reviews with no analyzable content."""
        processing_time_ms = int((time.time() - start_time) * 1000)

        return ReviewAnalysisResult(
            review_id=review_id,
            app_id=app_id,
            original_text=review_text,
            processed_text="",
            aspects=[],
            processing_time_ms=processing_time_ms,
            processing_model=self.processing_model,
            processing_version=self.processing_version,
            analysis_metadata={'status': 'empty_after_preprocessing'}
        )

    def _create_error_result(self, review_id: str, app_id: str, review_text: str, start_time: float, error: str) -> ReviewAnalysisResult:
        """Create error result for failed analysis."""
        processing_time_ms = int((time.time() - start_time) * 1000)

        return ReviewAnalysisResult(
            review_id=review_id,
            app_id=app_id,
            original_text=review_text,
            processed_text="",
            aspects=[],
            processing_time_ms=processing_time_ms,
            processing_model=self.processing_model,
            processing_version=self.processing_version,
            analysis_metadata={'status': 'error', 'error_message': error}
        )

    def get_analysis_summary(self, results: List[ReviewAnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics for a batch of analysis results."""
        if not results:
            return {}

        total_reviews = len(results)
        successful_reviews = len([r for r in results if r.aspects])
        empty_reviews = len([r for r in results if not r.aspects and
                           r.analysis_metadata.get('status') != 'error'])
        error_reviews = len([r for r in results if
                           r.analysis_metadata.get('status') == 'error'])

        # Aspect statistics
        all_aspects = []
        all_sentiments = []
        all_confidences = []

        for result in results:
            for aspect in result.aspects:
                all_aspects.append(aspect.aspect)
                all_sentiments.append(aspect.sentiment_score)
                all_confidences.append(aspect.confidence_score)

        # Calculate processing statistics
        processing_times = [r.processing_time_ms for r in results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        # Count aspect frequency
        aspect_counts = {}
        for aspect in all_aspects:
            aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

        return {
            'total_reviews': total_reviews,
            'successful_reviews': successful_reviews,
            'empty_reviews': empty_reviews,
            'error_reviews': error_reviews,
            'total_aspects_extracted': len(all_aspects),
            'unique_aspects': len(set(all_aspects)),
            'avg_aspects_per_review': len(all_aspects) / total_reviews if total_reviews > 0 else 0,
            'avg_sentiment_score': np.mean(all_sentiments) if all_sentiments else 0,
            'avg_confidence_score': np.mean(all_confidences) if all_confidences else 0,
            'avg_processing_time_ms': avg_processing_time,
            'most_common_aspects': sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'processing_model': self.processing_model,
            'batch_size': self.batch_size
        }

    def convert_to_database_format(self, results: List[ReviewAnalysisResult]) -> List[Dict[str, Any]]:
        """
        Convert analysis results to format suitable for deep_absa database table.

        Returns:
            List of dictionaries ready for database insertion
        """
        database_records = []

        for result in results:
            for aspect in result.aspects:
                record = {
                    'review_id': result.review_id,
                    'app_id': result.app_id,
                    'aspect': aspect.aspect,
                    'sentiment_score': aspect.sentiment_score,
                    'confidence_score': aspect.confidence_score,
                    'opinion_text': aspect.opinion_text,
                    'opinion_start_pos': aspect.opinion_start_pos,
                    'opinion_end_pos': aspect.opinion_end_pos,
                    'processing_model': result.processing_model,
                    'processing_version': result.processing_version
                }
                database_records.append(record)

        return database_records


# Convenience functions for easy usage
def analyze_review(review_id: str, app_id: str, review_text: str) -> ReviewAnalysisResult:
    """Convenience function to analyze a single review."""
    engine = DeepABSAEngine()
    return engine.analyze_single_review(review_id, app_id, review_text)


def analyze_review_batch(reviews: List[Dict[str, str]]) -> List[ReviewAnalysisResult]:
    """Convenience function to analyze a batch of reviews."""
    engine = DeepABSAEngine()
    return engine.analyze_batch(reviews)