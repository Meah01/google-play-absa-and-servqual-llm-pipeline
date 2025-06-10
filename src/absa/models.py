"""
Model management for ABSA Pipeline.
Handles loading, caching, and memory management of ML models for both deep and quick ABSA analysis.
Optimized for memory-constrained local development with model swapping capabilities.
"""

from __future__ import annotations

import os
import gc
import logging
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import torch
import spacy
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    pipeline
)

from src.utils.config import config


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_name: str
    model_type: str  # 'sentiment', 'aspect', 'spacy'
    model_size_mb: float
    loaded_at: datetime
    last_used: datetime
    usage_count: int


@dataclass
class ModelBundle:
    """Bundle containing model and tokenizer."""
    model: Any
    tokenizer: Any
    info: ModelInfo


class MemoryMonitor:
    """Monitors system memory usage for model management."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.models.memory")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent,
                'free_gb': memory.free / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {}

    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in MB based on model name."""
        size_estimates = {
            # Sentiment models
            'cardiffnlp/twitter-roberta-base-sentiment-latest': 500,
            'cardiffnlp/twitter-roberta-base-sentiment': 500,
            'distilbert-base-uncased-finetuned-sst-2-english': 250,
            'nlptown/bert-base-multilingual-uncased-sentiment': 700,

            # General models
            'distilbert-base-uncased': 250,
            'roberta-base': 500,
            'bert-base-uncased': 440,

            # spaCy models
            'en_core_web_sm': 15,
            'en_core_web_md': 50,
            'en_core_web_lg': 750
        }

        # Default estimate for unknown models
        default_size = 300

        return size_estimates.get(model_name, default_size)

    def can_load_model(self, model_name: str, safety_margin_gb: float = 1.0) -> bool:
        """Check if there's enough memory to load a model."""
        try:
            memory_info = self.get_memory_usage()
            available_gb = memory_info.get('available_gb', 0)
            estimated_size_gb = self.estimate_model_size(model_name) / 1024

            required_memory = estimated_size_gb + safety_margin_gb

            self.logger.debug(f"Memory check for {model_name}: "
                            f"Available: {available_gb:.2f}GB, "
                            f"Required: {required_memory:.2f}GB")

            return available_gb >= required_memory

        except Exception as e:
            self.logger.error(f"Error checking memory for model loading: {e}")
            return True  # Default to allowing load


class ModelCache:
    """Manages model caching with memory constraints."""

    def __init__(self, max_cache_size_gb: float = 2.0):
        self.logger = logging.getLogger("absa_pipeline.models.cache")
        self.max_cache_size_gb = max_cache_size_gb
        self.cached_models: Dict[str, ModelBundle] = {}
        self.memory_monitor = MemoryMonitor()

    def get_cache_size_mb(self) -> float:
        """Calculate current cache size in MB."""
        return sum(bundle.info.model_size_mb for bundle in self.cached_models.values())

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_size_mb = self.get_cache_size_mb()
        return {
            'cached_models': len(self.cached_models),
            'cache_size_mb': cache_size_mb,
            'cache_size_gb': cache_size_mb / 1024,
            'max_cache_size_gb': self.max_cache_size_gb,
            'models': [bundle.info for bundle in self.cached_models.values()]
        }

    def make_space_for_model(self, model_name: str) -> bool:
        """Free up space for a new model by removing least recently used models."""
        required_size_mb = self.memory_monitor.estimate_model_size(model_name)
        current_size_mb = self.get_cache_size_mb()
        max_size_mb = self.max_cache_size_gb * 1024

        # Check if we need to free space
        if current_size_mb + required_size_mb <= max_size_mb:
            return True

        self.logger.info(f"Making space for {model_name} ({required_size_mb}MB)")

        # Sort models by last used time (oldest first)
        models_by_usage = sorted(
            self.cached_models.items(),
            key=lambda x: x[1].info.last_used
        )

        # Remove models until we have enough space
        for model_key, bundle in models_by_usage:
            if current_size_mb + required_size_mb <= max_size_mb:
                break

            self.logger.info(f"Removing {model_key} from cache to free memory")
            self.remove_model(model_key)
            current_size_mb = self.get_cache_size_mb()

        return current_size_mb + required_size_mb <= max_size_mb

    def add_model(self, model_key: str, bundle: ModelBundle) -> bool:
        """Add a model to the cache."""
        try:
            if self.make_space_for_model(model_key):
                self.cached_models[model_key] = bundle
                self.logger.info(f"Added {model_key} to cache")
                return True
            else:
                self.logger.warning(f"Cannot add {model_key} to cache - insufficient space")
                return False
        except Exception as e:
            self.logger.error(f"Error adding model to cache: {e}")
            return False

    def get_model(self, model_key: str) -> Optional[ModelBundle]:
        """Get a model from cache and update usage statistics."""
        if model_key in self.cached_models:
            bundle = self.cached_models[model_key]
            bundle.info.last_used = datetime.now()
            bundle.info.usage_count += 1
            self.logger.debug(f"Retrieved {model_key} from cache")
            return bundle
        return None

    def remove_model(self, model_key: str) -> bool:
        """Remove a model from cache and free memory."""
        if model_key in self.cached_models:
            try:
                bundle = self.cached_models.pop(model_key)

                # Clear model from memory
                if hasattr(bundle.model, 'cpu'):
                    bundle.model.cpu()
                del bundle.model
                del bundle.tokenizer

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.logger.info(f"Removed {model_key} from cache")
                return True

            except Exception as e:
                self.logger.error(f"Error removing model from cache: {e}")
                return False

        return False

    def clear_cache(self) -> None:
        """Clear all cached models."""
        model_keys = list(self.cached_models.keys())
        for model_key in model_keys:
            self.remove_model(model_key)

        self.logger.info("Cache cleared")


class ModelLoader:
    """Loads and manages individual models."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.models.loader")
        self.memory_monitor = MemoryMonitor()

    def load_sentiment_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load sentiment analysis model and tokenizer."""
        try:
            self.logger.info(f"Loading sentiment model: {model_name}")

            # Check memory before loading
            if not self.memory_monitor.can_load_model(model_name):
                raise MemoryError(f"Insufficient memory to load {model_name}")

            # Load tokenizer and model
            if 'roberta' in model_name.lower():
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                model = RobertaForSequenceClassification.from_pretrained(model_name)
            elif 'distilbert' in model_name.lower():
                tokenizer = DistilBertTokenizer.from_pretrained(model_name)
                model = DistilBertForSequenceClassification.from_pretrained(model_name)
            else:
                # Use AutoTokenizer/AutoModel for other models
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Set model to evaluation mode
            model.eval()

            self.logger.info(f"Successfully loaded sentiment model: {model_name}")
            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Error loading sentiment model {model_name}: {e}")
            raise

    def load_spacy_model(self, model_name: str = "en_core_web_md") -> Any:
        """Load spaCy model for linguistic processing."""
        try:
            self.logger.info(f"Loading spaCy model: {model_name}")

            # Try to load the requested model
            try:
                nlp = spacy.load(model_name)
                self.logger.info(f"Successfully loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                # Fallback to smaller model
                if model_name != "en_core_web_sm":
                    self.logger.warning(f"Failed to load {model_name}, trying en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                    self.logger.info("Successfully loaded fallback spaCy model: en_core_web_sm")
                    return nlp
                else:
                    raise

        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {e}")
            self.logger.warning("Creating basic English model without pre-trained vectors")
            # Create basic English model as last resort
            from spacy.lang.en import English
            return English()

    def create_sentiment_pipeline(self, model: Any, tokenizer: Any) -> Any:
        """Create a HuggingFace pipeline for sentiment analysis."""
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
                truncation=True,
                max_length=config.absa.max_text_length
            )

            self.logger.debug("Created sentiment analysis pipeline")
            return sentiment_pipeline

        except Exception as e:
            self.logger.error(f"Error creating sentiment pipeline: {e}")
            raise


class ModelManager:
    """Main model management class that coordinates loading, caching, and memory management."""

    def __init__(self, max_cache_size_gb: float = 2.0):
        self.logger = logging.getLogger("absa_pipeline.models.manager")
        self.cache = ModelCache(max_cache_size_gb)
        self.loader = ModelLoader()
        self.memory_monitor = MemoryMonitor()

        # Current model configuration from config
        self.deep_model_name = config.absa.deep_model_name
        self.quick_model_name = config.absa.quick_model_name
        self.spacy_model_name = config.absa.aspect_extraction_model

        self.logger.info("ModelManager initialized")

    def get_model_key(self, model_name: str, model_type: str) -> str:
        """Generate cache key for a model."""
        return f"{model_type}:{model_name}"

    def load_deep_absa_models(self) -> Tuple[Any, Any, Any]:
        """Load models for deep ABSA analysis (RoBERTa + spaCy)."""
        self.logger.info("Loading deep ABSA models")

        # Load sentiment model (RoBERTa)
        sentiment_key = self.get_model_key(self.deep_model_name, "sentiment")
        sentiment_bundle = self.cache.get_model(sentiment_key)

        if sentiment_bundle is None:
            model, tokenizer = self.loader.load_sentiment_model(self.deep_model_name)

            # Create model info
            model_info = ModelInfo(
                model_name=self.deep_model_name,
                model_type="sentiment",
                model_size_mb=self.memory_monitor.estimate_model_size(self.deep_model_name),
                loaded_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=1
            )

            sentiment_bundle = ModelBundle(model=model, tokenizer=tokenizer, info=model_info)
            self.cache.add_model(sentiment_key, sentiment_bundle)

        # Load spaCy model for aspect extraction
        spacy_key = self.get_model_key(self.spacy_model_name, "spacy")
        spacy_bundle = self.cache.get_model(spacy_key)

        if spacy_bundle is None:
            nlp = self.loader.load_spacy_model(self.spacy_model_name)

            model_info = ModelInfo(
                model_name=self.spacy_model_name,
                model_type="spacy",
                model_size_mb=self.memory_monitor.estimate_model_size(self.spacy_model_name),
                loaded_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=1
            )

            spacy_bundle = ModelBundle(model=nlp, tokenizer=None, info=model_info)
            self.cache.add_model(spacy_key, spacy_bundle)

        return (
            sentiment_bundle.model,
            sentiment_bundle.tokenizer,
            spacy_bundle.model
        )

    def load_quick_absa_models(self) -> Tuple[Any, Any]:
        """Load models for quick ABSA analysis (DistilBERT). Phase 3 implementation."""
        self.logger.info("Loading quick ABSA models")

        quick_key = self.get_model_key(self.quick_model_name, "quick_sentiment")
        quick_bundle = self.cache.get_model(quick_key)

        if quick_bundle is None:
            model, tokenizer = self.loader.load_sentiment_model(self.quick_model_name)

            model_info = ModelInfo(
                model_name=self.quick_model_name,
                model_type="quick_sentiment",
                model_size_mb=self.memory_monitor.estimate_model_size(self.quick_model_name),
                loaded_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=1
            )

            quick_bundle = ModelBundle(model=model, tokenizer=tokenizer, info=model_info)
            self.cache.add_model(quick_key, quick_bundle)

        return quick_bundle.model, quick_bundle.tokenizer

    def create_sentiment_pipeline(self, model_type: str = "deep") -> Any:
        """Create a sentiment analysis pipeline."""
        try:
            if model_type == "deep":
                model, tokenizer, _ = self.load_deep_absa_models()
            elif model_type == "quick":
                model, tokenizer = self.load_quick_absa_models()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            return self.loader.create_sentiment_pipeline(model, tokenizer)

        except Exception as e:
            self.logger.error(f"Error creating {model_type} sentiment pipeline: {e}")
            raise

    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory and cache status."""
        memory_info = self.memory_monitor.get_memory_usage()
        cache_info = self.cache.get_cache_info()

        return {
            'memory': memory_info,
            'cache': cache_info,
            'model_config': {
                'deep_model': self.deep_model_name,
                'quick_model': self.quick_model_name,
                'spacy_model': self.spacy_model_name
            }
        }

    def clear_models(self) -> None:
        """Clear all cached models and free memory."""
        self.logger.info("Clearing all models")
        self.cache.clear_cache()

    def preload_models(self, model_types: List[str] = None) -> bool:
        """Preload specified models for faster inference."""
        if model_types is None:
            model_types = ["deep"]  # Default to deep models only

        try:
            for model_type in model_types:
                if model_type == "deep":
                    self.load_deep_absa_models()
                elif model_type == "quick":
                    self.load_quick_absa_models()
                else:
                    self.logger.warning(f"Unknown model type for preloading: {model_type}")

            self.logger.info(f"Successfully preloaded models: {model_types}")
            return True

        except Exception as e:
            self.logger.error(f"Error preloading models: {e}")
            return False


# Global model manager instance
model_manager = ModelManager()


# Convenience functions for easy usage
def get_deep_absa_models() -> Tuple[Any, Any, Any]:
    """Convenience function to get deep ABSA models."""
    return model_manager.load_deep_absa_models()


def get_quick_absa_models() -> Tuple[Any, Any]:
    """Convenience function to get quick ABSA models."""
    return model_manager.load_quick_absa_models()


def create_sentiment_pipeline(model_type: str = "deep") -> Any:
    """Convenience function to create sentiment pipeline."""
    return model_manager.create_sentiment_pipeline(model_type)


def get_memory_status() -> Dict[str, Any]:
    """Convenience function to get memory status."""
    return model_manager.get_memory_status()


def clear_all_models() -> None:
    """Convenience function to clear all models."""
    model_manager.clear_models()