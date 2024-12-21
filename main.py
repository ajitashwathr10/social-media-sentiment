import tweepy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprecessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, request, jsonify
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import gensim
from gensim.models import Word2Vec
import logging
import os
from datetime import datetime
import json

class SentimentAnalyzer:
    def __init__(self, max_features = 10000, max_len = 150):
        nltk.download('punkt', quiet = True)
        nltk.download('stopwords', quiet = True)
        nltk.download('vader_lexicon', quiet = True)

        self.max_features = max_features
        self.max_len = max_len

        self.tokenizer = Tokenizer(num_words = max_features, oov_token = '<OOV>')
        self.label_encoder = LabelEncoder()
        self.word2vec_model = None
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_intensity = SentimentIntensityAnalyzer()

        logging.basicConfig(
            filename = 'sentiment_analysis.log',
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s: %(message)s'
        )
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
            self.max_features,
            256,
            input_length = self.max_len
            mask_zero = True
            ),
            tf.keras.layers.SpatialDropout1D(0.3),
            tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                return_sequqences = True,
                dropout = 0.2,
                recurrent_dropout = 0.2
            )
            ),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(256, activation = 'relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                5,
                activation = 'softmax'
            )
        ])

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model
    
    def _text_cleaning(self, text):
        text = text.lower()
        text = re.sub(r'https\S+|www\S+|https\S+', '', text, flags = re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        doc = self.nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha and len(token.lemma_) > 1
        ]
        return ' '.join(tokens)

    def _train_model(
        self,
        X_train,
        y_train,
        epochs = 25,
        batch_size = 128,
        validation_split = 0.2
    ):
        """
        Training with techniques to avoid overfitting

        Args:
            X_train (np.array): Training sequences
            y_train (np.array): Training labels
            epochs (int): Maximum training iterations
            batch_size (int): Training batch size
            validation_split (float): Validation data ratio
        """

        y_train_categorical = tf.keras.utils.to_categorical(
            self.label_encoder.fit_transform(y_train)
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 8,
            restore_best_weight = True
        )
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.3,
            patience = 4,
            min_lr = 1e-6,
            verbose = 1
        )
        history = self.model.fit(
            X_train,
            y_train_categorical,
            epochs = epochs,
            batch_size = batch_size,
            validation_split = validation_split,
            callbacks = [early_stop, lr_reduce]
        )
        self.log_training_performance(history)
    
    def _log_training_performance(self, history):
        """
        Comprehensive training performance logging
        
        Args:
            history (tf.keras.callbacks.History): Training history
        """

        log_file = 'model_training_log.json'
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1],
            'precision': history.history['precision'][-1],
            'recall': history.history['recall'][-1]
        }

        with open(log_file, 'a') as f:
            json.dump(performance_data, f)
            f.write('\n')
    
    def _train_word2vec(self, texts):
        """
        Train word embeddings for semantic understanding

        Args:
            texts (list): Collection of processed texts
        """
        tokenized_texts = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(
            sentences = tokenized_texts,
            vector_size = 300,
            window = 5,
            min_count = 1
        )

    def _predict_sentiment(self, tweets, use_ensemble = True):
        """
        Multi-modal sentiment prediction
        
        Args:
            tweets (list): Input tweet texts
            use_ensemble (bool): Enable ensemble prediction
        
        Returns:
            list: Comprehensive sentiment analysis results
        """

        processed_tweets = [self._text_cleaning(tweet) for tweet in tweets]
        self.tokenizer.fit_on_texts(processed_tweets)
        sequences = self.tokenizer.texts_to_sequences(processed_tweets)
        padded_sequences = pad_sequences(sequences, maxlen = self.max_len)
        nn_predictions = self.model.predict(padded_sequences)
        results = []
        for tweet, nn_pred, processed_text in zip(tweets, nn_predictions, processed_tweets):
            vadar_scores = self.sentiment_intensity.polarity_scores(tweet)
            semantic_features = {}
            if self.word2vec_model:
                semantic_features = self._extract_semantic_features(processed_text)
            sentiment_labels = [
                'Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'
            ]
            if use_ensemble:
                nn_sentiment = sentiment_labels[np.argmax(nn_pred)]
                result = {
                    'text': tweet,
                    'neural_network_sentiment': nn_sentiment,
                    'neural_network_probabilities': dict(zip(sentiment_labels, nn_pred)),
                    'vadar_sentiment': vadar_scores,
                    'semantic_features': semantic_features
                }
            else:
                result = {
                    'text': tweet,
                    'neural_network_sentiment': sentiment_labels[np.argmax(nn_pred)],
                    'neural_network_probabilities': dict(zip(sentiment_labels, nn_pred))
                }
            results.append(result)
        return results

    def _extract_semantic_features(self, text):
        """
        Extract semantic features using Word2Vec
        
        Args:
            text (str): Processed text
            
        Returns:
            dict: Semantic analysis results
        """

        if not self.word2vec_model:
            return {}
        tokens = text.split()
        semantic_vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv
        ]
        if not semantic_vectors:
            return {}
        return {
            'avg_semantic_vector': np.mean(semantic_vectors, axis = 0).tolist(),
            'semantic_vector_variance': np.var(semantic_vectors, axis = 0).tolist()
        }

class TwitterDataCollector:
    def __init__(self, consumer_key, consumer_secret, access_token ,acces_token_secret):
        try:
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set.access_token(access_token, acces_token_secret)
            self.api = tweepy.API(auth, wait_on_rate_limit = True)
            self.client = tweepy.Client(
                bearer_token = consumer_secret,
                wait_on_rate_limit = True
            )
            self.logger = self._setup_logger()
        except Exception as e:
            logging.error(f"Twitter Authentication Error: {e}")
            raise
    
    def _setup_logger(self):
        logger = logging.getLogger('twitter_collector')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('twitter_collection.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _collect_tweets(self, query, count = 1000, lang = 'en', include_metrics = True):
        tweets_data = []
        try:
            for tweet in tweepy.Cursor(
                self.api.search_tweets,
                q = query,
                lang = lang
                tweet_mode = 'extended',
                include_entities = True
            ).items(count):
                tweet_data = {
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'user_followers': tweet.user.followers_count,
                    'user_location': tweet.user.location,
                    'user_verified': tweet.user.verified,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'hashtags': [tag['text'] for tag in tweet.entities.get('hashtags', [])],
                    'mentions': [mention['screen_name'] for mention in tweet.entities.get('user_mentions', [])],
                    'source': tweet.source,
                    'is_retweet': hasattr(tweet, 'retweeted_status')
                }    
                if include_metrics:
                    tweets_data.update(self._get_engagement_metrics(tweet))
                tweets_data.append(tweet_data)
                self.logger.info(f"Collected tweet from {tweet.user.screen_name}")

        except Exception as e:
            self.logger.error(f"Error collecting tweets: {e}")
        return pd.DateFrame(tweets_data)

    def _get_engagement_metrics(self, tweet):
        metrics = {}
        try:
            potential_reach = tweet.user.followers_count
            if potential_reach > 0:
                engagement_rate = ((tweet.favorite_count + tweet.retweet_count) / potential_reach) * 100
                metrics['engagement_rate'] = round(engagement_rate, 2)
            time_diff = datetime.now(tweet.created_at.tzinfo) - tweet.created_at
            hours_passed = time_diff.total_seconds() / 3600
            if hours_passed > 0:
                virality_score = (tweet.retweet_count / hours_passed) * 100
                metrics['virality_score'] = round(virality_score, 2)
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
        return metrics

class TweetAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.tweet_collector = None
        self.setup_routes()
        self.setup_error_handling()

        self.request_count = {}
        self.rate_limit = 100
        self.cache = {}
        self.cache_timeout = 300

    def setup_routes(self):
        @self.app.route('/api/v1/analyze', methods = ['POST'])
        def analyze_sentiment():
            if not self._check_rate_limit(request.remote_addr):
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            


