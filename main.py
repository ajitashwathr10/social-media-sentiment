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
        
    