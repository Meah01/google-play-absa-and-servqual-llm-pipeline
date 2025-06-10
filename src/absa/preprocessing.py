"""
Text preprocessing for ABSA Pipeline.
Handles text cleaning, normalization, and preparation for both deep and quick ABSA analysis.
Optimized for ecommerce app review analysis with emoji handling and aspect keyword preparation.
"""

from __future__ import annotations

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
import pandas as pd
import spacy
from spacy.lang.en import English

from src.utils.config import config


@dataclass
class PreprocessingResult:
    """Result of text preprocessing operation."""
    original_text: str
    cleaned_text: str
    deep_processed_text: str
    quick_processed_text: str
    detected_aspects: List[str]
    text_stats: Dict[str, Any]
    preprocessing_flags: Dict[str, bool]


class EmojiHandler:
    """Handles emoji processing for review text."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.preprocessing.emoji")

        # Common emoji to text mappings for ecommerce reviews
        self.emoji_mappings = {
            # Sentiment emojis
            '😀': ' happy ',
            '😁': ' happy ',
            '😂': ' funny ',
            '😊': ' happy ',
            '😍': ' love ',
            '😎': ' cool ',
            '😔': ' sad ',
            '😞': ' disappointed ',
            '😠': ' angry ',
            '😡': ' angry ',
            '🙄': ' annoyed ',

            # Thumbs and hands
            '👍': ' good ',
            '👎': ' bad ',
            '👌': ' perfect ',
            '👏': ' great ',
            '🤝': ' good ',
            '✋': ' stop ',

            # Hearts and rating
            '❤️': ' love ',
            '💕': ' love ',
            '💖': ' love ',
            '💯': ' perfect ',
            '⭐': ' star ',
            '🌟': ' star ',

            # Objects relevant to ecommerce
            '📦': ' package ',
            '🚚': ' delivery ',
            '💰': ' money ',
            '💳': ' payment ',
            '🛒': ' shopping ',
            '🛍️': ' shopping ',
            '📱': ' phone ',
            '💻': ' computer ',

            # Check marks and crosses
            '✅': ' good ',
            '❌': ' bad ',
            '✔️': ' good ',
            '❎': ' bad ',
            '🚫': ' bad ',

            # Time and speed
            '⏰': ' time ',
            '⚡': ' fast ',
            '🐌': ' slow ',
            '⏳': ' waiting ',

            # Quality indicators
            '🔥': ' great ',
            '💎': ' premium ',
            '🗑️': ' trash ',
            '💩': ' bad '
        }

    def convert_emojis_to_text(self, text: str) -> str:
        """Convert emojis to descriptive text."""
        processed_text = text

        for emoji, replacement in self.emoji_mappings.items():
            processed_text = processed_text.replace(emoji, replacement)

        return processed_text

    def remove_unmapped_emojis(self, text: str) -> str:
        """Remove emojis that weren't converted to text."""
        try:
            # Simplified approach - remove common emoji ranges one by one
            # This avoids complex regex character set issues
            patterns = [
                r'[\U0001F600-\U0001F64F]+',  # emoticons
                r'[\U0001F300-\U0001F5FF]+',  # symbols & pictographs
                r'[\U0001F680-\U0001F6FF]+',  # transport & map symbols
                r'[\U0001F1E0-\U0001F1FF]+',  # flags
                r'[\U00002702-\U000027B0]+',  # misc symbols
                r'[\U000024C2-\U0001F251]+',  # enclosed characters
            ]

            # Apply each pattern separately
            for pattern in patterns:
                text = re.sub(pattern, ' ', text)

            return text

        except Exception as e:
            self.logger.error(f"Error removing emojis: {e}")
            # If all else fails, return text as-is
            return text

    def process_emojis(self, text: str, convert_to_text: bool = True) -> str:
        """Main emoji processing function - temporarily simplified."""
        try:
            if convert_to_text:
                # Convert known emojis to text
                text = self.convert_emojis_to_text(text)

            # Skip complex emoji removal for now
            return text

        except Exception as e:
            self.logger.error(f"Error processing emojis: {e}")
            return text


class TextCleaner:
    """Core text cleaning functionality."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.preprocessing.cleaner")

        # Contractions mapping for proper expansion
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "here's": "here is",
            "there's": "there is",
            "where's": "where is",
            "how's": "how is",
            "let's": "let us",
            "who's": "who is",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "mustn't": "must not"
        }

    def expand_contractions(self, text: str) -> str:
        """Expand contractions for better analysis."""
        text_lower = text.lower()

        for contraction, expansion in self.contractions.items():
            text_lower = text_lower.replace(contraction, expansion)

        return text_lower

    def clean_special_characters(self, text: str) -> str:
        """Clean special characters while preserving meaning."""
        try:
            # Replace multiple dots with single period
            text = re.sub(r'\.{2,}', '.', text)

            # Replace multiple exclamation marks with single
            text = re.sub(r'!{2,}', '!', text)

            # Replace multiple question marks with single
            text = re.sub(r'\?{2,}', '?', text)

            # Remove excessive punctuation combinations
            text = re.sub(r'[!?]{3,}', '!', text)

            # Clean up quotation marks - FIXED: Use escape sequences instead of smart quotes
            text = re.sub(r'[\u201c\u201d]', '"', text)  # Smart double quotes
            text = re.sub(r'[\u2018\u2019]', "'", text)  # Smart single quotes

            # Replace multiple dashes with single
            text = re.sub(r'-{2,}', '-', text)

            # Clean up asterisks (often used for emphasis)
            text = re.sub(r'\*+', ' ', text)

            return text

        except Exception as e:
            self.logger.error(f"Error cleaning special characters: {e}")
            return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and remove excessive spacing."""
        # Replace tabs and newlines with spaces
        text = re.sub(r'[\t\n\r]', ' ', text)

        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)

        # Strip leading and trailing whitespace
        text = text.strip()

        return text

    def handle_repetitive_characters(self, text: str) -> str:
        """Handle repetitive characters (e.g., 'sooooo good' -> 'so good')."""
        # Reduce repetitive characters to maximum of 2
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        return text

    def clean_urls_and_mentions(self, text: str) -> str:
        """Remove URLs and mentions that don't add sentiment value."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove mentions (less common in app reviews but good to handle)
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)

        return text

    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """Main text cleaning function."""
        if not text or not isinstance(text, str):
            return ""

        # Expand contractions first
        text = self.expand_contractions(text)

        # Clean URLs and mentions
        text = self.clean_urls_and_mentions(text)

        # Handle repetitive characters
        text = self.handle_repetitive_characters(text)

        # Clean special characters
        text = self.clean_special_characters(text)

        # Normalize whitespace
        text = self.normalize_whitespace(text)

        # Convert to lowercase unless preserving case
        if not preserve_case:
            text = text.lower()

        return text


class AspectKeywordExtractor:
    """Extracts potential aspect keywords from text for preprocessing."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.preprocessing.aspects")
        self.nlp = None
        self._load_spacy_model()
        self._load_aspect_keywords()

    def _load_spacy_model(self):
        """Load spaCy model for linguistic processing."""
        try:
            self.nlp = spacy.load("en_core_web_md")
            self.logger.info("Loaded spaCy model: en_core_web_md")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.warning("Fallback to en_core_web_sm model")
            except OSError:
                # Create a basic English model if no spaCy models available
                self.nlp = English()
                self.logger.warning("Using basic English model - aspect extraction will be limited")

    def _load_aspect_keywords(self):
        """Load aspect keywords from database."""
        try:
            from src.data.storage import storage

            query = """
            SELECT aspect_name, keywords, servqual_dimension
            FROM aspects 
            WHERE is_active = TRUE
            """

            df = storage.db.execute_query(query)

            self.aspect_keywords = {}
            self.aspect_to_dimension = {}

            for _, row in df.iterrows():
                aspect_name = row['aspect_name']
                keywords = row['keywords'] if row['keywords'] else []
                dimension = row['servqual_dimension']

                self.aspect_keywords[aspect_name] = keywords
                if dimension:
                    self.aspect_to_dimension[aspect_name] = dimension

            self.logger.info(f"Loaded {len(self.aspect_keywords)} aspect keyword sets")

        except Exception as e:
            self.logger.error(f"Failed to load aspect keywords: {e}")
            # Fallback to basic keywords
            self.aspect_keywords = {
                'product_quality': ['quality', 'cheap', 'durable'],
                'shipping_delivery': ['shipping', 'delivery', 'fast'],
                'customer_service': ['service', 'support', 'help']
            }
            self.aspect_to_dimension = {}

    def extract_potential_aspects(self, text: str) -> List[str]:
        """Extract potential aspects mentioned in the text."""
        detected_aspects = []
        text_lower = text.lower()

        for aspect_name, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected_aspects.append(aspect_name)
                    break  # Only count each aspect once per text

        return detected_aspects

    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases that might be aspects."""
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text)
            noun_phrases = []

            for chunk in doc.noun_chunks:
                # Filter out pronouns and very short phrases
                if len(chunk.text) > 2 and chunk.root.pos_ == 'NOUN':
                    noun_phrases.append(chunk.text.lower().strip())

            return noun_phrases

        except Exception as e:
            self.logger.error(f"Error extracting noun phrases: {e}")
            return []


class ABSAPreprocessor:
    """Main preprocessing class that orchestrates all text processing for ABSA."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.preprocessing")

        # Initialize components
        self.emoji_handler = EmojiHandler()
        self.text_cleaner = TextCleaner()
        self.aspect_extractor = AspectKeywordExtractor()

        # Configuration
        self.max_text_length = config.absa.max_text_length

        self.logger.info("ABSA Preprocessor initialized")

    def get_text_statistics(self, original: str, cleaned: str) -> Dict[str, Any]:
        """Generate text processing statistics."""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'word_count': len(cleaned.split()),
            'sentence_count': len([s for s in cleaned.split('.') if s.strip()]),
            'compression_ratio': len(cleaned) / len(original) if len(original) > 0 else 0
        }

    def preprocess_for_deep_analysis(self, text: str) -> str:
        """Preprocessing optimized for deep ABSA analysis (RoBERTa)."""
        # Convert emojis to text for semantic understanding
        text = self.emoji_handler.process_emojis(text, convert_to_text=True)

        # Clean text while preserving meaning
        text = self.text_cleaner.clean_text(text, preserve_case=False)

        # Truncate if too long for transformer models
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]

        return text

    def preprocess_for_quick_analysis(self, text: str) -> str:
        """Preprocessing optimized for quick ABSA analysis (keyword-based)."""
        # Remove emojis for cleaner keyword matching
        text = self.emoji_handler.process_emojis(text, convert_to_text=False)

        # More aggressive cleaning for keyword matching
        text = self.text_cleaner.clean_text(text, preserve_case=False)

        return text

    def preprocess_single_review(self, text: str) -> PreprocessingResult:
        """Preprocess a single review text with complete analysis."""
        if not text or not isinstance(text, str):
            return self._create_empty_result(text)

        try:
            original_text = text

            # Basic cleaning first
            cleaned_text = self.text_cleaner.clean_text(text, preserve_case=True)

            # Generate different versions for different analysis types
            deep_processed = self.preprocess_for_deep_analysis(original_text)
            quick_processed = self.preprocess_for_quick_analysis(original_text)

            # Extract potential aspects
            detected_aspects = self.aspect_extractor.extract_potential_aspects(cleaned_text)

            # Generate statistics
            text_stats = self.get_text_statistics(original_text, cleaned_text)

            # Set preprocessing flags
            preprocessing_flags = {
                'has_emojis': any(char in original_text for char in self.emoji_handler.emoji_mappings.keys()),
                'has_aspects': len(detected_aspects) > 0,
                'is_truncated': len(deep_processed) < len(original_text),
                'is_empty_after_cleaning': len(cleaned_text.strip()) == 0
            }

            return PreprocessingResult(
                original_text=original_text,
                cleaned_text=cleaned_text,
                deep_processed_text=deep_processed,
                quick_processed_text=quick_processed,
                detected_aspects=detected_aspects,
                text_stats=text_stats,
                preprocessing_flags=preprocessing_flags
            )

        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return self._create_empty_result(text)

    def preprocess_batch(self, texts: List[str]) -> List[PreprocessingResult]:
        """Preprocess a batch of review texts efficiently."""
        results = []

        for i, text in enumerate(texts):
            try:
                result = self.preprocess_single_review(text)
                results.append(result)

                if (i + 1) % 10 == 0:
                    self.logger.debug(f"Preprocessed {i + 1}/{len(texts)} reviews")

            except Exception as e:
                self.logger.error(f"Error preprocessing review {i}: {e}")
                results.append(self._create_empty_result(text))

        self.logger.info(f"Batch preprocessing complete: {len(results)} reviews processed")
        return results

    def _create_empty_result(self, original_text: str) -> PreprocessingResult:
        """Create empty preprocessing result for error cases."""
        return PreprocessingResult(
            original_text=original_text or "",
            cleaned_text="",
            deep_processed_text="",
            quick_processed_text="",
            detected_aspects=[],
            text_stats={'original_length': 0, 'cleaned_length': 0, 'word_count': 0, 'sentence_count': 0, 'compression_ratio': 0},
            preprocessing_flags={'has_emojis': False, 'has_aspects': False, 'is_truncated': False, 'is_empty_after_cleaning': True}
        )

    def get_preprocessing_summary(self, results: List[PreprocessingResult]) -> Dict[str, Any]:
        """Generate summary statistics for a batch of preprocessing results."""
        if not results:
            return {}

        total_reviews = len(results)
        empty_after_cleaning = sum(1 for r in results if r.preprocessing_flags['is_empty_after_cleaning'])
        has_emojis = sum(1 for r in results if r.preprocessing_flags['has_emojis'])
        has_aspects = sum(1 for r in results if r.preprocessing_flags['has_aspects'])
        is_truncated = sum(1 for r in results if r.preprocessing_flags['is_truncated'])

        avg_compression = sum(r.text_stats['compression_ratio'] for r in results) / total_reviews
        avg_word_count = sum(r.text_stats['word_count'] for r in results) / total_reviews

        # Count aspect frequency
        aspect_counts = {}
        for result in results:
            for aspect in result.detected_aspects:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

        return {
            'total_reviews': total_reviews,
            'empty_after_cleaning': empty_after_cleaning,
            'reviews_with_emojis': has_emojis,
            'reviews_with_aspects': has_aspects,
            'truncated_reviews': is_truncated,
            'avg_compression_ratio': round(avg_compression, 3),
            'avg_word_count': round(avg_word_count, 1),
            'most_common_aspects': sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }


# Convenience functions for easy usage
def preprocess_review_text(text: str) -> PreprocessingResult:
    """Convenience function to preprocess a single review."""
    preprocessor = ABSAPreprocessor()
    return preprocessor.preprocess_single_review(text)


def preprocess_review_batch(texts: List[str]) -> List[PreprocessingResult]:
    """Convenience function to preprocess a batch of reviews."""
    preprocessor = ABSAPreprocessor()
    return preprocessor.preprocess_batch(texts)