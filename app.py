from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import tweepy
import re
import os
from datetime import datetime
import pandas as pd
from collections import Counter

app = Flask(__name__)

TWITTER_API_KEY = ""
TWITTER_API_SECRET = ""
TWITTER_ACCESS_TOKEN = ""
TWITTER_ACCESS_TOKEN_SECRET = ""
