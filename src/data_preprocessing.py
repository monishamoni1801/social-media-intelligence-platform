import pandas as pd
import numpy as np
import re
from datetime import datetime
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download these once if not already
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class FeatureEngineer:
    def __init__(self, filepath='../data/social_media_data.csv'):
        self.df = pd.read_csv(filepath)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Industry-grade text cleaning"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags (we extract them separately)
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def extract_hashtag_features(self):
        """Extract features from hashtags"""
        def count_hashtags(tag_str):
            if pd.isna(tag_str):
                return 0
            return len(str(tag_str).split())

        def get_hashtag_length(tag_str):
            if pd.isna(tag_str):
                return 0
            return sum(len(tag) for tag in str(tag_str).split())

        self.df['num_hashtags'] = self.df['Hashtags'].apply(count_hashtags)
        self.df['avg_hashtag_length'] = self.df['Hashtags'].apply(get_hashtag_length)
        self.df['avg_hashtag_length'] = np.where(
            self.df['num_hashtags'] > 0,
            self.df['avg_hashtag_length'] / self.df['num_hashtags'],
            0
        )
        return self

    def extract_engagement_features(self):
        """Create engagement rate features"""
        self.df['engagement_rate'] = (
            (self.df['Likes'] + self.df['Retweets']) / 
            (self.df['Likes'].max() + self.df['Retweets'].max())
        )
        self.df['like_retweet_ratio'] = np.where(
            self.df['Retweets'] > 0,
            self.df['Likes'] / (self.df['Retweets'] + 1),
            0
        )
        return self

    def extract_sentiment_features(self):
        """Generate sentiment scores using TextBlob and VADER"""
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        sia = SentimentIntensityAnalyzer()
        
        # Clean text for sentiment
        self.df['cleaned_text'] = self.df['Text'].apply(self.clean_text)
        
        # TextBlob sentiment
        self.df['textblob_polarity'] = self.df['cleaned_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        self.df['textblob_subjectivity'] = self.df['cleaned_text'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        
        # VADER sentiment for social media (handles caps, emojis, etc. better)
        self.df['vader_compound'] = self.df['Text'].apply(
            lambda x: sia.polarity_scores(str(x))['compound']
        )
        
        # Use VADER for final sentiment classification
        self.df['vader_sentiment'] = self.df['vader_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        return self

    def extract_temporal_features(self):
        """Extract time-based features for trend prediction"""
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['hour'] = self.df['Timestamp'].dt.hour
        self.df['day_of_week'] = self.df['Timestamp'].dt.dayofweek
        self.df['month'] = self.df['Timestamp'].dt.month
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (so 23 and 0 are close)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        return self

    def create_trend_label(self):
        """Create target variable: Will this post trend?"""
        # Define trending as top 20% by engagement
        engagement_score = self.df['Likes'] + self.df['Retweets'] * 2  # Retweets weighted higher
        threshold = engagement_score.quantile(0.8)
        self.df['will_trend'] = (engagement_score > threshold).astype(int)
        return self

    def get_processed_data(self):
        """Execute all feature engineering steps"""
        self.extract_hashtag_features()
        self.extract_engagement_features()
        self.extract_sentiment_features()
        self.extract_temporal_features()
        self.create_trend_label()
        
        # Select final features for modeling
        feature_columns = [
            'num_hashtags', 'avg_hashtag_length', 'engagement_rate',
            'like_retweet_ratio', 'textblob_polarity', 'vader_compound',
            'hour_sin', 'hour_cos', 'is_weekend', 'month'
        ]
        
        X = self.df[feature_columns]
        y = self.df['will_trend']
        
        return X, y, self.df

# Run it
if __name__ == "__main__":
    engineer = FeatureEngineer('../data/social_media_data.csv')
    X, y, df_processed = engineer.get_processed_data()
    print(f"Data processed! Shape: {X.shape}")
    print(f"Trending posts: {y.sum()} out of {len(y)} ({y.mean()*100:.1f}%)")
    df_processed.to_csv('../data/processed_data.csv', index=False)