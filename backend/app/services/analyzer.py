"""
Enhanced Review analysis service with AI/NLP capabilities.
Includes sentiment analysis, emotion detection, and topic modeling.
UPDATED: Added better error handling, optional imports, and lightweight fallbacks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from collections import Counter
from datetime import datetime
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import hashlib
from functools import lru_cache

# ADDED: Optional imports for AI libraries
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    print("⚠️ Transformers not available - using lightweight models only")

try:
    import text2emotion as te
    TEXT2EMOTION_AVAILABLE = True
    print("✅ text2emotion library available")
except ImportError:
    TEXT2EMOTION_AVAILABLE = False
    print("⚠️ text2emotion not available - using basic emotion detection")

try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
    print("✅ NRCLex library available")
except ImportError:
    NRCLEX_AVAILABLE = False
    print("⚠️ NRCLex not available - using basic emotion detection")

from app.core.config import settings
from app.utils.text_cleaner import text_cleaner
from app.utils.helpers import sanitize_dataframe, calculate_percentage

class EnhancedReviewAnalyzer:
    """Advanced review analysis with AI/NLP capabilities."""
    
    def __init__(self):
        self.text_cleaner = text_cleaner
        
        # IMPROVED: Better NLTK data management
        self._ensure_nltk_data()
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # IMPROVED: Initialize AI models only if available
        self.device = -1  # Default to CPU for Railway deployment
        self.roberta_sentiment = None
        self.emotion_classifier = None
        
        if TRANSFORMERS_AVAILABLE:
            self.device = self._select_optimal_device()
            self.roberta_sentiment = self._load_roberta_model()
            self.emotion_classifier = self._load_emotion_model()
        
        # ADDED: Performance tracking
        self.model_load_success = {
            'roberta': self.roberta_sentiment is not None,
            'emotion': self.emotion_classifier is not None,
            'transformers': TRANSFORMERS_AVAILABLE,
            'text2emotion': TEXT2EMOTION_AVAILABLE,
            'nrclex': NRCLEX_AVAILABLE
        }
        
        print(f"✅ Enhanced Analyzer initialized")
        print(f"📊 Available models: {[k for k, v in self.model_load_success.items() if v]}")
    
    # ADDED: Better NLTK data management
    def _ensure_nltk_data(self):
        """Download NLTK data only if not present."""
        required_data = {
            'punkt': 'tokenizers/punkt',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'brown': 'corpora/brown',
            'vader_lexicon': 'vader_lexicon',
            'stopwords': 'corpora/stopwords'
        }
        
        for name, path in required_data.items():
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"📥 Downloading NLTK {name}...")
                try:
                    nltk.download(name, quiet=True)
                    print(f"✅ Downloaded {name}")
                except Exception as e:
                    print(f"⚠️ Failed to download NLTK {name}: {e}")
    
    # ADDED: Better device selection (CPU-focused for Railway)
    def _select_optimal_device(self):
        """Select optimal device - prioritize CPU for Railway."""
        if not TRANSFORMERS_AVAILABLE or not torch:
            return -1
        
        # For Railway deployment, prefer CPU to avoid memory issues
        use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'
        
        if use_gpu and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                test_tensor = torch.zeros(100, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ GPU available and working")
                return 0
            except Exception as e:
                print(f"⚠️ GPU available but unusable ({e}), using CPU")
                return -1
        else:
            print("✅ Using CPU for processing")
            return -1
    
    # IMPROVED: Better model loading with Railway-specific optimizations
    def _load_roberta_model(self):
        """Load RoBERTa with Railway optimizations."""
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        # Skip heavy models on Railway unless explicitly enabled
        enable_heavy_models = os.getenv('ENABLE_HEAVY_MODELS', 'false').lower() == 'true'
        if not enable_heavy_models:
            print("⚠️ Heavy models disabled for Railway deployment")
            return None
        
        try:
            model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device,
                return_all_scores=False
            )
            print("✅ RoBERTa sentiment model loaded")
            return model
        except Exception as e:
            print(f"⚠️ RoBERTa model failed to load: {e}")
            return None
    
    # IMPROVED: Better emotion model loading
    def _load_emotion_model(self):
        """Load emotion classifier with Railway optimizations."""
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        # Skip heavy models on Railway unless explicitly enabled
        enable_heavy_models = os.getenv('ENABLE_HEAVY_MODELS', 'false').lower() == 'true'
        if not enable_heavy_models:
            print("⚠️ Heavy emotion models disabled for Railway deployment")
            return None
        
        try:
            classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device,
                top_k=None
            )
            print("✅ Emotion detection model loaded")
            return classifier
        except Exception as e:
            print(f"⚠️ Emotion model failed to load: {e}")
            return None
    
    # ADDED: Comprehensive input validation
    def _validate_and_sanitize_input(self, reviews_data: Dict[str, Any]) -> Tuple[bool, str, List[Dict]]:
        """Comprehensive input validation and sanitization."""
        try:
            # Type validation
            if not isinstance(reviews_data, dict):
                return False, "Input must be a dictionary", []
            
            # Success flag validation
            if not reviews_data.get("success"):
                error = reviews_data.get("error", "Unknown error in input data")
                return False, f"Input data error: {error}", []
            
            # Reviews list validation
            reviews = reviews_data.get("reviews", [])
            if not isinstance(reviews, list):
                return False, "Reviews must be a list", []
            
            if not reviews:
                return False, "No reviews to analyze", []
            
            # ADDED: Sanitize and validate each review
            valid_reviews = []
            for i, review in enumerate(reviews):
                try:
                    sanitized = self._sanitize_single_review(review, i)
                    if sanitized:
                        valid_reviews.append(sanitized)
                except Exception as e:
                    print(f"⚠️ Failed to sanitize review {i}: {e}")
                    continue
            
            if not valid_reviews:
                return False, "No valid reviews found after sanitization", []
            
            print(f"✅ Validated {len(valid_reviews)}/{len(reviews)} reviews")
            return True, "Valid input", valid_reviews
        
        except Exception as e:
            return False, f"Validation failed: {str(e)}", []
    
    # ADDED: Individual review sanitization
    def _sanitize_single_review(self, review: Dict, index: int) -> Dict:
        """Sanitize and validate a single review."""
        try:
            if not isinstance(review, dict):
                return None
            
            # Essential fields validation
            review_text = review.get('review_text') or review.get('text', '')
            if not review_text or not isinstance(review_text, str):
                return None
            
            # Clean and validate text
            review_text = str(review_text).strip()
            if len(review_text) < 5:  # Minimum meaningful length
                return None
            
            # Sanitize rating
            rating = review.get('rating', 0)
            try:
                rating = float(rating)
                if not (0 <= rating <= 5):
                    rating = 3.0  # Default to neutral if invalid
            except (ValueError, TypeError):
                rating = 3.0
            
            # Build sanitized review
            sanitized = {
                'review_text': review_text,
                'rating': rating,
                'review_date': review.get('review_date', review.get('date')),
                'verified_purchase': bool(review.get('verified_purchase', False)),
                'helpful_votes': max(0, int(review.get('helpful_votes', 0))),
                'reviewer_name': review.get('reviewer_name', f'Reviewer_{index}')
            }
            
            return sanitized
        
        except Exception as e:
            print(f"⚠️ Review sanitization failed: {e}")
            return None
    
    def analyze_reviews(self, reviews_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analysis pipeline with AI/NLP features."""
        try:
            # IMPROVED: Comprehensive input validation
            is_valid, error_msg, validated_reviews = self._validate_and_sanitize_input(reviews_data)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "validation_error"
                }
            
            print(f"🔍 Starting enhanced analysis of {len(validated_reviews)} reviews")
            
            # IMPROVED: Better DataFrame preparation with error handling
            try:
                df = self._prepare_dataframe_enhanced(validated_reviews)
                if df.empty:
                    return {
                        "success": False,
                        "error": "Failed to create valid DataFrame from reviews",
                        "error_type": "dataframe_error"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"DataFrame preparation failed: {str(e)}",
                    "error_type": "dataframe_error"
                }
            
            # IMPROVED: Batch text cleaning for performance
            print("🧹 Cleaning text data...")
            df['cleaned_text'] = self._batch_clean_text(df['review_text'].tolist())
            
            # IMPROVED: Perform analyses with individual error handling
            results = {}
            analysis_functions = [
                ('sentiment_distribution', self._analyze_sentiment_lightweight),
                ('emotion_analysis', self._analyze_emotions_lightweight),
                ('keyword_analysis', self._extract_keywords_advanced),
                ('topic_modeling', self._perform_topic_modeling),
                ('rating_distribution', self._analyze_rating_distribution),
                ('temporal_trends', self._analyze_temporal_trends),
                ('quality_metrics', self._analyze_review_quality),
                ('customer_segments', self._segment_customers)
            ]
            
            for key, func in analysis_functions:
                try:
                    print(f"📊 Running {key}...")
                    results[key] = func(df)
                except Exception as e:
                    print(f"⚠️ {key} failed: {e}")
                    results[key] = {"error": str(e), "status": "failed"}
            
            # IMPROVED: Generate insights with error handling
            try:
                insights = self._generate_ai_insights(
                    df, results.get('sentiment_distribution', {}), 
                    results.get('emotion_analysis', {}),
                    results.get('keyword_analysis', {}), 
                    results.get('topic_modeling', {}), 
                    results.get('customer_segments', {})
                )
                results['insights'] = insights
            except Exception as e:
                print(f"⚠️ Insights generation failed: {e}")
                results['insights'] = [f"Insights generation failed: {str(e)}"]
            
            # IMPROVED: Generate summary with error handling
            try:
                summary = self._generate_executive_summary(
                    df, results.get('sentiment_distribution', {}), 
                    results.get('emotion_analysis', {}), 
                    results.get('insights', [])
                )
                results['summary'] = summary
            except Exception as e:
                print(f"⚠️ Summary generation failed: {e}")
                results['summary'] = f"Summary generation failed: {str(e)}"
            
            print(f"✅ Enhanced analysis completed successfully")
            
            # IMPROVED: Return comprehensive results
            return {
                "success": True,
                "asin": reviews_data.get("asin", ""),
                "product_title": reviews_data.get("product_info", {}).get("title", "Unknown Product"),
                "total_reviews": len(df),
                "analyzed_at": datetime.now().isoformat(),
                
                # Analysis results
                **results,
                
                # Metadata
                "api_source": reviews_data.get("api_source", "apify"),
                "max_reviews_limit": reviews_data.get("max_reviews_limit", 5),
                "ai_models_used": self._get_active_models(),
                "fallback_used": reviews_data.get("fallback", False),
                "analysis_version": "2.0_lightweight",  # UPDATED: Lightweight version
                "deployment_mode": "railway"
            }
            
        except Exception as e:
            print(f"❌ Critical analysis error: {e}")
            return {
                "success": False,
                "error": f"Critical analysis failure: {str(e)}",
                "error_type": "critical_error"
            }
    
    # IMPROVED: Enhanced DataFrame preparation
    def _prepare_dataframe_enhanced(self, reviews: List[Dict]) -> pd.DataFrame:
        """Enhanced DataFrame preparation with better validation."""
        try:
            if not reviews:
                return pd.DataFrame()
            
            df = pd.DataFrame(reviews)
            
            # IMPROVED: More robust data cleaning
            df = df.dropna(subset=['review_text'])  # Remove rows without text
            df = df[df['review_text'].str.len() > 5]  # Remove very short reviews
            
            # IMPROVED: Better type conversion with error handling
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0)
            df['rating'] = df['rating'].clip(1, 5)  # Ensure valid rating range
            
            # IMPROVED: Better date handling
            if 'review_date' in df.columns:
                df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
            
            # IMPROVED: Ensure required columns exist
            required_columns = ['review_text', 'rating']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            print(f"📋 DataFrame prepared: {len(df)} valid reviews")
            return df
            
        except Exception as e:
            print(f"❌ DataFrame preparation failed: {e}")
            raise
    
    # ADDED: Batch text cleaning for performance
    def _batch_clean_text(self, texts: List[str]) -> pd.Series:
        """Clean multiple texts efficiently."""
        try:
            cleaned_texts = []
            for text in texts:
                try:
                    cleaned = self.text_cleaner.clean_text(str(text) if text else '', remove_stopwords=False)
                    cleaned_texts.append(cleaned)
                except Exception as e:
                    print(f"⚠️ Text cleaning failed for one item: {e}")
                    cleaned_texts.append(str(text))  # Fallback to original
            
            return pd.Series(cleaned_texts)
        except Exception as e:
            print(f"⚠️ Batch text cleaning failed: {e}")
            return pd.Series([str(t) for t in texts])  # Fallback
    
    # IMPROVED: Lightweight sentiment analysis for Railway
    def _analyze_sentiment_lightweight(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Lightweight sentiment analysis using VADER and TextBlob only."""
        try:
            texts = df['cleaned_text'].tolist()
            
            if not texts:
                return self._empty_sentiment_result()
            
            print(f"🎯 Processing {len(texts)} texts for sentiment (lightweight mode)...")
            
            # Process with VADER and TextBlob only (no heavy AI models)
            sentiments = []
            
            for text in texts:
                try:
                    # VADER Sentiment
                    vader_scores = self.vader.polarity_scores(str(text))
                    
                    # TextBlob Sentiment
                    blob = TextBlob(str(text))
                    textblob_polarity = blob.sentiment.polarity
                    textblob_subjectivity = blob.sentiment.subjectivity
                    
                    # Combine scores (no RoBERTa in lightweight mode)
                    combined_score = (vader_scores['compound'] + textblob_polarity) / 2
                    
                    sentiments.append({
                        'combined_score': combined_score,
                        'vader_compound': vader_scores['compound'],
                        'textblob_polarity': textblob_polarity,
                        'textblob_subjectivity': textblob_subjectivity,
                        'roberta_score': None,  # Not available in lightweight mode
                        'label': self._get_sentiment_label(combined_score)
                    })
                    
                except Exception as e:
                    print(f"⚠️ Sentiment analysis failed for one text: {e}")
                    sentiments.append(self._default_sentiment())
            
            return self._format_sentiment_results(sentiments, df)
            
        except Exception as e:
            print(f"❌ Sentiment analysis failed: {e}")
            return self._empty_sentiment_result()
    
    # IMPROVED: Lightweight emotion analysis
    def _analyze_emotions_lightweight(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Lightweight emotion detection without heavy models."""
        try:
            texts = df['cleaned_text'].tolist()
            if not texts:
                return self._empty_emotion_result()
            
            print(f"😊 Processing {len(texts)} texts for emotions (lightweight mode)...")
            
            all_emotions = []
            emotion_distribution = Counter()
            
            # Use text2emotion if available, otherwise basic keyword detection
            for text in texts:
                try:
                    if TEXT2EMOTION_AVAILABLE:
                        emotions = te.get_emotion(str(text))
                        if emotions:
                            max_emotion = max(emotions, key=emotions.get)
                            if emotions[max_emotion] > 0:
                                all_emotions.append({
                                    'text': str(text)[:100],
                                    'emotion': max_emotion,
                                    'score': emotions[max_emotion]
                                })
                                emotion_distribution[max_emotion] += 1
                    else:
                        # Basic keyword-based emotion detection
                        emotion = self._basic_emotion_detection(str(text))
                        if emotion:
                            all_emotions.append({
                                'text': str(text)[:100],
                                'emotion': emotion,
                                'score': 0.5
                            })
                            emotion_distribution[emotion] += 1
                            
                except Exception as e:
                    print(f"⚠️ Emotion detection failed for one text: {e}")
                    continue
            
            return self._format_emotion_results(emotion_distribution, all_emotions)
            
        except Exception as e:
            print(f"❌ Emotion analysis failed: {e}")
            return self._empty_emotion_result()
    
    # ADDED: Basic keyword-based emotion detection
    def _basic_emotion_detection(self, text: str) -> str:
        """Basic emotion detection using keywords."""
        text_lower = text.lower()
        
        # Simple keyword mapping
        emotion_keywords = {
            'joy': ['happy', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love'],
            'anger': ['angry', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'frustrated'],
            'sadness': ['sad', 'disappointed', 'poor', 'bad', 'unhappy', 'regret'],
            'surprise': ['surprised', 'unexpected', 'wow', 'amazing', 'incredible'],
            'trust': ['reliable', 'trustworthy', 'quality', 'recommend', 'solid'],
            'fear': ['worried', 'concerned', 'afraid', 'risky', 'dangerous']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        return 'neutral'
    
    # Keep all the helper methods with the same signatures...
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """Return empty sentiment result structure."""
        return {
            "positive": {"count": 0, "percentage": 0},
            "neutral": {"count": 0, "percentage": 0},
            "negative": {"count": 0, "percentage": 0},
            "average_rating": 0,
            "median_rating": 0,
            "sentiment_scores": {
                "average_compound": 0,
                "average_subjectivity": 0.5,
                "confidence": 0
            }
        }
    
    def _empty_emotion_result(self) -> Dict[str, Any]:
        """Return empty emotion result structure."""
        return {
            "emotion_distribution": {},
            "dominant_emotions": [],
            "emotional_tone": "neutral",
            "emotion_samples": [],
            "total_emotions_detected": 0
        }
    
    def _default_sentiment(self) -> Dict[str, Any]:
        """Return default sentiment for failed cases."""
        return {
            'combined_score': 0,
            'vader_compound': 0,
            'textblob_polarity': 0,
            'textblob_subjectivity': 0.5,
            'roberta_score': None,
            'label': 'neutral'
        }
    
    def _format_sentiment_results(self, sentiments: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
        """Format sentiment analysis results."""
        try:
            total = len(sentiments)
            if total == 0:
                return self._empty_sentiment_result()
            
            # Count labels
            positive = sum(1 for s in sentiments if s['label'] == 'positive')
            negative = sum(1 for s in sentiments if s['label'] == 'negative')
            neutral = total - positive - negative
            
            # Calculate confidence scores
            subjectivity_scores = [s['textblob_subjectivity'] for s in sentiments if s['textblob_subjectivity'] is not None]
            avg_subjectivity = np.mean(subjectivity_scores) if subjectivity_scores else 0.5
            sentiment_confidence = max(0, 1 - avg_subjectivity)
            
            return {
                "positive": {
                    "count": int(positive),
                    "percentage": calculate_percentage(positive, total)
                },
                "neutral": {
                    "count": int(neutral),
                    "percentage": calculate_percentage(neutral, total)
                },
                "negative": {
                    "count": int(negative),
                    "percentage": calculate_percentage(negative, total)
                },
                "average_rating": round(float(df['rating'].mean()), 2),
                "median_rating": round(float(df['rating'].median()), 2),
                "sentiment_scores": {
                    "average_compound": round(float(np.mean([s['combined_score'] for s in sentiments])), 3),
                    "average_subjectivity": round(float(avg_subjectivity), 3),
                    "confidence": round(float(sentiment_confidence), 3)
                }
            }
        except Exception as e:
            print(f"⚠️ Sentiment result formatting failed: {e}")
            return self._empty_sentiment_result()
    
    def _format_emotion_results(self, emotion_distribution: Counter, all_emotions: List[Dict]) -> Dict[str, Any]:
        """Format emotion analysis results."""
        try:
            # Calculate percentages
            total_emotions = sum(emotion_distribution.values())
            emotion_percentages = {}
            
            if total_emotions > 0:
                for emotion, count in emotion_distribution.most_common():
                    emotion_percentages[emotion] = {
                        'count': float(count),
                        'percentage': round((count / total_emotions) * 100, 2)
                    }
            
            # Identify dominant emotions
            dominant_emotions = []
            for emotion, data in list(emotion_percentages.items())[:3]:
                dominant_emotions.append({
                    'emotion': emotion,
                    'percentage': data['percentage'],
                    'description': self._get_emotion_description(emotion)
                })
            
            return {
                "emotion_distribution": emotion_percentages,
                "dominant_emotions": dominant_emotions,
                "emotional_tone": self._determine_emotional_tone(emotion_distribution),
                "emotion_samples": all_emotions[:5],
                "total_emotions_detected": len(all_emotions)
            }
        except Exception as e:
            print(f"⚠️ Emotion result formatting failed: {e}")
            return self._empty_emotion_result()
    
    # Continue with all existing methods but with improved error handling...
    def _extract_keywords_advanced(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract keywords with TF-IDF and improved error handling."""
        try:
            return self._extract_keywords_tfidf(df)
        except Exception as e:
            print(f"⚠️ Keyword extraction failed: {e}")
            return {"top_keywords": [], "total_unique_words": 0, "error": str(e)}
    
    def _extract_keywords_tfidf(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract keywords using TF-IDF with improved error handling."""
        try:
            valid_texts = df['cleaned_text'][df['cleaned_text'].str.len() > 10].tolist()
            
            if not valid_texts:
                return {"top_keywords": [], "total_unique_words": 0}
            
            vectorizer = TfidfVectorizer(
                max_features=min(100, len(valid_texts) * 10),
                ngram_range=(1, 2),
                min_df=max(1, len(valid_texts) // 10),
                max_df=0.7,
                stop_words='english',
                token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()
            avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            keyword_scores = list(zip(feature_names, avg_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_keywords = []
            max_keywords = getattr(settings, 'TOP_KEYWORDS_COUNT', 20)
            
            for word, score in keyword_scores[:max_keywords]:
                try:
                    frequency = sum(1 for text in df['cleaned_text'] if word in str(text).lower())
                    
                    top_keywords.append({
                        "word": word,
                        "tfidf_score": round(float(score), 4),
                        "frequency": int(frequency),
                        "importance": "high" if score > 0.1 else "medium" if score > 0.05 else "low"
                    })
                except Exception as e:
                    print(f"⚠️ Keyword processing failed for '{word}': {e}")
                    continue
            
            return {
                "top_keywords": top_keywords,
                "total_unique_words": int(len(feature_names))
            }
            
        except Exception as e:
            print(f"❌ TF-IDF keyword extraction failed: {e}")
            return {"top_keywords": [], "total_unique_words": 0, "error": str(e)}
    
    def _perform_topic_modeling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform topic modeling with improved error handling."""
        try:
            valid_texts = df['cleaned_text'][df['cleaned_text'].str.len() > 20].tolist()
            
            if len(valid_texts) < 3:
                return {"topics": [], "message": "Insufficient data for topic modeling (need at least 3 reviews)"}
            
            vectorizer = TfidfVectorizer(
                max_features=min(50, len(valid_texts) * 5),
                min_df=max(1, len(valid_texts) // 10),
                max_df=0.8,
                stop_words='english',
                ngram_range=(1, 2),
                token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
            
            doc_term_matrix = vectorizer.fit_transform(valid_texts)
            
            n_topics = min(max(2, len(valid_texts) // 3), 5)
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='batch'
            )
            
            lda.fit(doc_term_matrix)
            
            # Extract topics with better error handling
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                try:
                    top_indices = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_indices]
                    topic_weight = float(topic[top_indices].mean())
                    
                    topics.append({
                        "topic_id": topic_idx,
                        "keywords": top_words[:5],
                        "weight": topic_weight,
                        "theme": self._infer_topic_theme(top_words)
                    })
                except Exception as e:
                    print(f"⚠️ Topic {topic_idx} processing failed: {e}")
                    continue
            
            return {
                "topics": topics,
                "num_topics": n_topics,
                "model": "LDA",
                "perplexity": float(lda.perplexity(doc_term_matrix)) if hasattr(lda, 'perplexity') else None
            }
            
        except Exception as e:
            print(f"❌ Topic modeling failed: {e}")
            return {"topics": [], "error": str(e)}
    
    # All remaining methods with improved error handling...
    def _segment_customers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Segment customers with error handling."""
        try:
            segments = {
                "enthusiasts": {"criteria": "5-star ratings with detailed positive reviews", "count": 0, "characteristics": []},
                "critics": {"criteria": "Low ratings with specific complaints", "count": 0, "characteristics": []},
                "pragmatists": {"criteria": "Balanced reviews with pros and cons", "count": 0, "characteristics": []},
                "casual": {"criteria": "Brief reviews without strong opinions", "count": 0, "characteristics": []}
            }
            
            for _, row in df.iterrows():
                try:
                    rating = float(row['rating'])
                    text_length = len(str(row['review_text']))
                    
                    if rating >= 4.5 and text_length > 100:
                        segments["enthusiasts"]["count"] += 1
                    elif rating <= 2.5 and text_length > 50:
                        segments["critics"]["count"] += 1
                    elif 3 <= rating <= 4 and text_length > 75:
                        segments["pragmatists"]["count"] += 1
                    else:
                        segments["casual"]["count"] += 1
                except Exception:
                    segments["casual"]["count"] += 1  # Default to casual
            
            # Calculate percentages
            total = len(df)
            for segment in segments.values():
                segment["percentage"] = calculate_percentage(segment["count"], total)
            
            return segments
            
        except Exception as e:
            print(f"❌ Customer segmentation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_review_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze review quality with error handling."""
        try:
            quality_scores = []
            
            for _, row in df.iterrows():
                try:
                    text = str(row['review_text'])
                    score = 0
                    factors = []
                    
                    # Length factor
                    if len(text) > 50:
                        score += 2
                        factors.append("detailed")
                    if len(text) > 200:
                        score += 1
                        factors.append("comprehensive")
                    
                    # Specificity factor
                    if any(word in text.lower() for word in ['because', 'specifically', 'particularly']):
                        score += 2
                        factors.append("specific")
                    
                    # Verified purchase
                    if row.get('verified_purchase', False):
                        score += 3
                        factors.append("verified")
                    
                    # Helpfulness
                    helpful_votes = int(row.get('helpful_votes', 0))
                    if helpful_votes > 5:
                        score += 2
                        factors.append("helpful")
                    
                    quality_scores.append({
                        'score': min(score, 10),
                        'factors': factors
                    })
                    
                except Exception:
                    quality_scores.append({'score': 3, 'factors': []})  # Default score
            
            if not quality_scores:
                return {"error": "No quality scores calculated"}
            
            avg_quality = np.mean([q['score'] for q in quality_scores])
            
            # Calculate verified percentage safely
            verified_count = sum(1 for _, row in df.iterrows() if row.get('verified_purchase', False))
            
            return {
                "average_quality_score": round(float(avg_quality), 2),
                "quality_distribution": {
                    "high": sum(1 for q in quality_scores if q['score'] >= 7),
                    "medium": sum(1 for q in quality_scores if 4 <= q['score'] < 7),
                    "low": sum(1 for q in quality_scores if q['score'] < 4)
                },
                "verified_percentage": calculate_percentage(verified_count, len(df)),
                "quality_factors": dict(Counter([f for q in quality_scores for f in q['factors']]))
            }
            
        except Exception as e:
            print(f"❌ Quality analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_rating_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get star rating distribution with error handling."""
        try:
            if df.empty:
                return {"5_star": 0, "4_star": 0, "3_star": 0, "2_star": 0, "1_star": 0}
            
            ratings = df['rating'].fillna(0)
            
            return {
                "5_star": int(len(ratings[ratings == 5.0])),
                "4_star": int(len(ratings[ratings == 4.0])),
                "3_star": int(len(ratings[ratings == 3.0])),
                "2_star": int(len(ratings[ratings == 2.0])),
                "1_star": int(len(ratings[ratings == 1.0]))
            }
        except Exception as e:
            print(f"❌ Rating distribution analysis failed: {e}")
            return {"5_star": 0, "4_star": 0, "3_star": 0, "2_star": 0, "1_star": 0}
    
    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends over time with error handling."""
        try:
            if 'review_date' not in df.columns or df.empty:
                return {"monthly_data": [], "trend": "unknown", "message": "No date information available"}
            
            df_temp = df.copy()
            df_temp['review_date'] = pd.to_datetime(df_temp['review_date'], errors='coerce')
            df_temp = df_temp.dropna(subset=['review_date']).sort_values('review_date')
            
            if df_temp.empty:
                return {"monthly_data": [], "trend": "unknown", "message": "No valid dates found"}
            
            df_temp['year_month'] = df_temp['review_date'].dt.to_period('M')
            monthly_counts = df_temp.groupby('year_month').size()
            monthly_avg_rating = df_temp.groupby('year_month')['rating'].mean()
            
            monthly_data = []
            for period in monthly_counts.index:
                try:
                    monthly_data.append({
                        "month": str(period),
                        "review_count": int(monthly_counts[period]),
                        "average_rating": round(float(monthly_avg_rating[period]), 2)
                    })
                except Exception:
                    continue
            
            # Determine trend
            trend = "stable"
            if len(monthly_data) > 1:
                try:
                    first_count = monthly_data[0]["review_count"]
                    last_count = monthly_data[-1]["review_count"]
                    if last_count > first_count * 1.2:
                        trend = "increasing"
                    elif last_count < first_count * 0.8:
                        trend = "decreasing"
                except:
                    trend = "unknown"
            
            return {
                "monthly_data": monthly_data[-12:],  # Last 12 months
                "trend": trend
            }
            
        except Exception as e:
            print(f"❌ Temporal analysis failed: {e}")
            return {"monthly_data": [], "trend": "unknown", "error": str(e)}
    
    def _generate_ai_insights(self, df, sentiment, emotions, keywords, topics, segments) -> List[str]:
        """Generate insights with error handling."""
        try:
            insights = []
            
            pos_pct = sentiment.get('positive', {}).get('percentage', 0)
            neg_pct = sentiment.get('negative', {}).get('percentage', 0)
            
            # Sentiment insights
            if pos_pct > 80:
                insights.append(f"🌟 Exceptional satisfaction rate ({pos_pct}%) indicates product excellence")
            elif pos_pct > 60:
                insights.append(f"✅ Strong positive reception ({pos_pct}%) shows product meets expectations")
            elif neg_pct > 40:
                insights.append(f"⚠️ High dissatisfaction ({neg_pct}%) requires immediate attention")
            
            # Emotion insights
            dominant_emotions = emotions.get('dominant_emotions', [])
            if dominant_emotions:
                top_emotion = dominant_emotions[0]
                emotion_name = top_emotion.get('emotion', 'unknown')
                if emotion_name in ['joy', 'happiness', 'excitement']:
                    insights.append(f"😊 Customers express {emotion_name} - excellent emotional connection")
                elif emotion_name in ['anger', 'disgust', 'disappointment']:
                    insights.append(f"😔 Prevalent {emotion_name} indicates customer frustration")
            
            # Topic insights
            topics_list = topics.get('topics', [])
            if topics_list:
                main_themes = [t.get('theme', 'Unknown') for t in topics_list[:2]]
                insights.append(f"🎯 Key discussion themes: {', '.join(main_themes)}")
            
            # Keyword insights
            top_keywords = keywords.get('top_keywords', [])
            if top_keywords:
                trending_keywords = [kw.get('word', '') for kw in top_keywords[:3]]
                trending_keywords = [k for k in trending_keywords if k]
                if trending_keywords:
                    insights.append(f"🔍 Trending topics: {', '.join(trending_keywords)}")
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            print(f"❌ Insights generation failed: {e}")
            return [f"Insights generation encountered an error: {str(e)}"]
    
    def _generate_executive_summary(self, df, sentiment, emotions, insights) -> str:
        """Generate executive summary with error handling."""
        try:
            total = len(df)
            avg_rating = sentiment.get('average_rating', 0)
            pos_pct = sentiment.get('positive', {}).get('percentage', 0)
            
            summary_parts = [
                f"Analysis of {total} customer reviews reveals ",
                f"{pos_pct}% positive sentiment with {avg_rating:.1f} stars average."
            ]
            
            if insights and len(insights) > 0:
                clean_insight = insights[0].replace('🌟', '').replace('✅', '').replace('⚠️', '').strip()
                if clean_insight:
                    summary_parts.append(f" {clean_insight}")
            
            return ''.join(summary_parts)
            
        except Exception as e:
            return f"Executive summary could not be generated: {str(e)}"
    
    # Helper methods
    def _get_sentiment_label(self, score: float) -> str:
        """Determine sentiment label from score."""
        try:
            if score > 0.1:
                return 'positive'
            elif score < -0.1:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _get_emotion_description(self, emotion: str) -> str:
        """Get description for emotion."""
        descriptions = {
            'joy': 'Customers are delighted with the product',
            'trust': 'High confidence in product quality',
            'fear': 'Concerns about product reliability',
            'surprise': 'Unexpected product experience',
            'sadness': 'Disappointment with purchase',
            'disgust': 'Strong negative reaction',
            'anger': 'Frustration with product or service',
            'anticipation': 'Excitement about the product',
            'neutral': 'Balanced customer feedback'
        }
        return descriptions.get(str(emotion).lower(), f'Customers express {emotion}')
    
    def _determine_emotional_tone(self, emotion_distribution: Counter) -> str:
        """Determine overall emotional tone."""
        try:
            if not emotion_distribution:
                return "neutral"
            
            positive_emotions = ['joy', 'trust', 'anticipation', 'love', 'happiness']
            negative_emotions = ['fear', 'sadness', 'disgust', 'anger', 'disappointment']
            
            positive_score = sum(emotion_distribution.get(e, 0) for e in positive_emotions)
            negative_score = sum(emotion_distribution.get(e, 0) for e in negative_emotions)
            
            if positive_score > negative_score * 2:
                return "very positive"
            elif positive_score > negative_score:
                return "positive"
            elif negative_score > positive_score * 2:
                return "very negative"
            elif negative_score > positive_score:
                return "negative"
            else:
                return "mixed"
        except:
            return "neutral"
    
    def _infer_topic_theme(self, keywords: List[str]) -> str:
        """Infer theme from topic keywords."""
        try:
            theme_patterns = {
                'Quality': ['quality', 'durable', 'build', 'material', 'solid'],
                'Price/Value': ['price', 'value', 'worth', 'money', 'expensive', 'cheap'],
                'Performance': ['work', 'performance', 'fast', 'speed', 'efficient'],
                'Design': ['design', 'look', 'color', 'style', 'beautiful'],
                'Usability': ['easy', 'simple', 'use', 'setup', 'install'],
                'Service': ['delivery', 'shipping', 'customer', 'service', 'support'],
                'Features': ['feature', 'function', 'capability', 'option']
            }
            
            keywords_text = ' '.join(str(k) for k in keywords).lower()
            
            for theme, patterns in theme_patterns.items():
                if any(pattern in keywords_text for pattern in patterns):
                    return theme
            
            return 'General Feedback'
        except:
            return 'General Feedback'
    
    def _get_active_models(self) -> List[str]:
        """Get list of active models."""
        try:
            models = ['VADER', 'TextBlob', 'NLTK', 'TF-IDF', 'LDA']
            
            if TRANSFORMERS_AVAILABLE and self.roberta_sentiment:
                models.append('RoBERTa')
            if TRANSFORMERS_AVAILABLE and self.emotion_classifier:
                models.append('DistilRoBERTa-Emotion')
            if TEXT2EMOTION_AVAILABLE:
                models.append('text2emotion')
            if NRCLEX_AVAILABLE:
                models.append('NRCLex')
            
            return models
        except:
            return ['Basic-NLP']


# Singleton instance
review_analyzer = EnhancedReviewAnalyzer()
