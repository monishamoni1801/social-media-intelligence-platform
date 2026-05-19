import pandas as pd
import re 
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

class TrendAnalyzer:
    def __init__(self, data_path='../data/processed_data.csv'):
        # Handle path correctly
        if not os.path.exists(data_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_data.csv')
            if os.path.exists(alt_path):
                data_path = alt_path
            else:
                raise FileNotFoundError(f"Data file not found at {data_path} or {alt_path}")
        
        self.df = pd.read_csv(data_path)
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler()
        
        if 'Timestamp' in self.df.columns:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        
        self._cache = {}
        self.model = None
        self.ml_scaler = None
        self._load_models_if_exist()

    def _load_models_if_exist(self):
        try:
            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(current_dir)
            
            model_path = os.path.join(project_root, 'models', 'xgboost.pkl')
            scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
            
            print(f"Looking for model at: {model_path}")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.ml_scaler = joblib.load(scaler_path)
                print("✅ ML models loaded successfully!")
            else:
                print("⚠️ ML models not found. Using rule-based prediction.")
        except Exception as e:
            print(f"⚠️ Could not load ML models: {e}")
            self.model = None
            self.ml_scaler = None

    def analyze_hashtag_trends(self, top_n=20, time_window_days=30):
        cache_key = f'hashtag_trends_{top_n}_{time_window_days}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        recent_date = self.df['Timestamp'].max() - timedelta(days=time_window_days)
        recent_df = self.df[self.df['Timestamp'] >= recent_date]
        
        hashtag_performance = []
        for idx, row in recent_df.iterrows():
            if pd.notna(row.get('Hashtags', '')):
                hashtags = str(row['Hashtags']).split()
                for ht in hashtags:
                    engagement_score = row.get('engagement_rate', 0) * 100
                    sentiment_score = row.get('vader_compound', 0)
                    performance = {
                        'hashtag': ht.lower().replace('#', ''),
                        'total_engagement': engagement_score,
                        'avg_sentiment': sentiment_score,
                        'will_trend': row.get('will_trend', 0),
                        'frequency': 1,
                        'recency': (self.df['Timestamp'].max() - row['Timestamp']).days
                    }
                    hashtag_performance.append(performance)
        
        hashtag_df = pd.DataFrame(hashtag_performance)
        
        if len(hashtag_df) == 0:
            return pd.DataFrame()
        
        trending_score = hashtag_df.groupby('hashtag').agg({
            'total_engagement': 'mean',
            'avg_sentiment': 'mean',
            'will_trend': 'sum',
            'frequency': 'count',
            'recency': 'mean'
        }).reset_index()
        
        trending_score['velocity'] = (
            trending_score['total_engagement'] * trending_score['frequency'] / 
            (trending_score['recency'] + 1)
        )
        
        trending_score['trend_probability'] = (
            trending_score['velocity'] * 0.4 +
            trending_score['total_engagement'] * 0.3 +
            (trending_score['avg_sentiment'] + 1) / 2 * 20 +
            trending_score['will_trend'] * 10
        ) / 100
        
        if len(trending_score) > 1:
            trending_score['trend_probability'] = self.scaler.fit_transform(
                trending_score[['trend_probability']]
            ).flatten()
        
        trending_score['growth_trend'] = np.where(
            trending_score['frequency'] > trending_score['frequency'].median(),
            '📈 Rising', '📉 Stable'
        )
        
        if trending_score['frequency'].max() > 0:
            trending_score['confidence'] = np.clip(
                trending_score['frequency'] / trending_score['frequency'].max() * 100, 0, 100
            )
        else:
            trending_score['confidence'] = 50
        
        result = trending_score.sort_values('trend_probability', ascending=False).head(top_n)
        self._cache[cache_key] = result
        return result

    def get_sentiment_trends(self, days=30):
        cache_key = f'sentiment_trends_{days}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.df['date'] = self.df['Timestamp'].dt.date
        daily_sentiment = self.df.groupby('date').agg({
            'vader_compound': ['mean', 'std', 'count'],
            'will_trend': 'mean',
            'engagement_rate': 'mean'
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 
                                   'post_count', 'trend_rate', 'engagement_rate']
        
        daily_sentiment['sentiment_ma7'] = daily_sentiment['sentiment_mean'].rolling(7).mean()
        daily_sentiment['sentiment_ma30'] = daily_sentiment['sentiment_mean'].rolling(30).mean()
        daily_sentiment['sentiment_trend'] = (
            daily_sentiment['sentiment_ma7'] - daily_sentiment['sentiment_ma30']
        )
        
        rolling_std = daily_sentiment['sentiment_mean'].rolling(30).std()
        rolling_mean = daily_sentiment['sentiment_mean'].rolling(30).mean()
        daily_sentiment['sentiment_zscore'] = (
            (daily_sentiment['sentiment_mean'] - rolling_mean) / rolling_std
        )
        daily_sentiment['is_anomaly'] = abs(daily_sentiment['sentiment_zscore']) > 2
        
        daily_sentiment['trend_direction'] = np.where(
            daily_sentiment['sentiment_trend'] > 0.1, '📈 Improving',
            np.where(daily_sentiment['sentiment_trend'] < -0.1, '📉 Declining', '➡️ Stable')
        )
        
        result = daily_sentiment.tail(days)
        self._cache[cache_key] = result
        return result

    def get_engagement_analytics(self):
        analytics = {
            'avg_likes': self.df['Likes'].mean(),
            'avg_retweets': self.df['Retweets'].mean(),
            'total_posts': len(self.df),
            'unique_hashtags': self.df['Hashtags'].nunique() if 'Hashtags' in self.df.columns else 0,
        }
        
        if 'hour' in self.df.columns:
            hour_engagement = self.df.groupby('hour')['engagement_rate'].agg(['mean', 'std'])
            analytics['engagement_by_hour'] = hour_engagement['mean'].to_dict()
            analytics['engagement_std_by_hour'] = hour_engagement['std'].to_dict()
            analytics['best_time_to_post'] = int(hour_engagement['mean'].idxmax()) if not hour_engagement['mean'].empty else 12
            analytics['worst_time_to_post'] = int(hour_engagement['mean'].idxmin()) if not hour_engagement['mean'].empty else 0
        else:
            analytics['engagement_by_hour'] = {}
            analytics['best_time_to_post'] = 12
        
        if 'day_of_week' in self.df.columns:
            dow_engagement = self.df.groupby('day_of_week')['engagement_rate'].mean()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            analytics['best_day'] = days[int(dow_engagement.idxmax())] if not dow_engagement.empty else 'Monday'
        else:
            analytics['best_day'] = 'Monday'
        
        if 'Platform' in self.df.columns:
            platform_stats = self.df.groupby('Platform').agg({
                'engagement_rate': 'mean',
                'Likes': 'mean',
                'Retweets': 'mean'
            }).round(2)
            analytics['platform_stats'] = platform_stats.to_dict('index')
        
        if 'vader_sentiment' in self.df.columns:
            sentiment_dist = self.df['vader_sentiment'].value_counts()
            analytics['sentiment_distribution'] = sentiment_dist.to_dict()
            analytics['dominant_sentiment'] = str(sentiment_dist.idxmax()) if not sentiment_dist.empty else 'neutral'
        else:
            analytics['sentiment_distribution'] = {'positive': 0, 'neutral': 0, 'negative': 0}
            analytics['dominant_sentiment'] = 'neutral'
        
        if 'engagement_rate' in self.df.columns:
            top_content = self.df.nlargest(10, 'engagement_rate')[
                ['Text', 'engagement_rate', 'Likes', 'Retweets', 'vader_sentiment', 'Hashtags']
            ].to_dict('records')
            analytics['top_performing_content'] = top_content
        else:
            analytics['top_performing_content'] = []
        
        analytics['engagement_trend'] = {
            'weekly_growth': self._calculate_growth_rate('engagement_rate'),
            'likes_growth': self._calculate_growth_rate('Likes'),
            'retweets_growth': self._calculate_growth_rate('Retweets')
        }
        
        analytics['recommendations'] = self._generate_recommendations()
        return analytics
    
    def _calculate_growth_rate(self, column):
        try:
            if column in self.df.columns and 'Timestamp' in self.df.columns:
                df_weekly = self.df.set_index('Timestamp').resample('W')[column].mean()
                if len(df_weekly) >= 2 and df_weekly.iloc[-2] != 0:
                    growth = (df_weekly.iloc[-1] - df_weekly.iloc[-2]) / abs(df_weekly.iloc[-2]) * 100
                    return round(growth, 1)
        except:
            pass
        return 0
    
    def _generate_recommendations(self):
        recommendations = []
        
        if 'hour' in self.df.columns and 'engagement_rate' in self.df.columns:
            best_hour = self.df.groupby('hour')['engagement_rate'].mean().idxmax()
            best_engagement = self.df[self.df['hour'] == best_hour]['engagement_rate'].mean() * 100
            recommendations.append({
                'type': 'timing', 'title': 'Optimal Posting Time',
                'message': f'Posts between {int(best_hour)}:00 - {int(best_hour)+1}:00 get {best_engagement:.1f}% more engagement',
                'icon': '⏰'
            })
        
        try:
            hashtag_performance = self.analyze_hashtag_trends(5)
            if len(hashtag_performance) > 0:
                top_hashtag = hashtag_performance.iloc[0]['hashtag']
                recommendations.append({
                    'type': 'hashtag', 'title': 'Trending Hashtag Alert',
                    'message': f'Use #{top_hashtag} for {hashtag_performance.iloc[0]["trend_probability"]*100:.0f}% higher reach',
                    'icon': '#️⃣'
                })
        except:
            pass
        
        if 'vader_compound' in self.df.columns:
            current_sentiment = self.df['vader_compound'].mean()
            if current_sentiment > 0.3:
                recommendations.append({
                    'type': 'sentiment', 'title': 'Positive Momentum',
                    'message': 'Audience sentiment is high! Capitalize with promotional content',
                    'icon': '😊'
                })
            elif current_sentiment < -0.2:
                recommendations.append({
                    'type': 'sentiment', 'title': 'Sentiment Alert',
                    'message': 'Negative sentiment detected. Consider addressing concerns',
                    'icon': '⚠️'
                })
        
        if 'vader_sentiment' in self.df.columns and 'engagement_rate' in self.df.columns:
            top_sentiment = self.df['vader_sentiment'].mode()
            if len(top_sentiment) > 0:
                top_sentiment = top_sentiment[0]
                sentiment_engagement = self.df[self.df['vader_sentiment'] == top_sentiment]['engagement_rate'].mean() * 100
                recommendations.append({
                    'type': 'content', 'title': 'Content Strategy',
                    'message': f'{top_sentiment.capitalize()} posts drive {sentiment_engagement:.0f}% more engagement',
                    'icon': '💡'
                })
        
        return recommendations

    def predict_viral_potential(self, text, hashtags):
        sentiment = self.sia.polarity_scores(text)['compound']
        text_length = len(text)
        excitement_words = ['amazing', 'incredible', 'awesome', 'excited', 'love', 'best', 'perfect', 'crazy', 'wow', '🚀', '🔥', '🎉']
        excitement_score = sum(word.lower() in text.lower() for word in excitement_words) / len(excitement_words)
        
        num_hashtags = len(hashtags.split()) if hashtags else 0
        avg_ht_length = sum(len(ht) for ht in hashtags.split()) / max(num_hashtags, 1)
        
        probability = None
        if self.model is not None and self.ml_scaler is not None:
            try:
                current_hour = datetime.now().hour
                current_weekday = datetime.now().weekday()
                
                avg_engagement = self.df['engagement_rate'].mean() if 'engagement_rate' in self.df.columns else 0.5
                avg_ratio = self.df['like_retweet_ratio'].mean() if 'like_retweet_ratio' in self.df.columns else 0.5
                avg_polarity = self.df['textblob_polarity'].mean() if 'textblob_polarity' in self.df.columns else 0.5
                
                features = np.array([[
                    num_hashtags, avg_ht_length, avg_engagement, avg_ratio, avg_polarity, sentiment,
                    np.sin(2 * np.pi * current_hour / 24),
                    np.cos(2 * np.pi * current_hour / 24),
                    1 if current_weekday >= 5 else 0,
                    datetime.now().month
                ]])
                
                features = np.nan_to_num(features)
                features_scaled = self.ml_scaler.transform(features)
                probability = self.model.predict_proba(features_scaled)[0, 1]
                print(f"ML Prediction: {probability:.3f}")
            except Exception as e:
                print(f"ML prediction failed: {e}")
                probability = None
        
        if probability is None:
            score = 0
            if 100 <= text_length <= 200:
                score += 30
            score += excitement_score * 30
            if 2 <= num_hashtags <= 5:
                score += 20
            if any(c in text for c in ['😊', '🔥', '🎉', '❤️', '💪', '🚀']):
                score += 15
            if '?' in text:
                score += 15
            if sentiment > 0.3:
                score += 20
            probability = min(0.95, max(0.05, score / 100))
        
        insights = []
        if text_length < 50:
            insights.append("📝 Short posts perform well on Twitter/X")
        elif text_length > 200:
            insights.append("📖 Longer content works better for engagement")
        else:
            insights.append("✅ Optimal length detected (100-200 chars)")
        
        if any(c in text for c in ['😊', '🔥', '🎉', '❤️', '💪', '🚀']):
            insights.append("😊 Emojis increase engagement by 25%")
        
        if excitement_score > 0.5:
            insights.append("⚡ High excitement language - great for viral potential")
        
        if num_hashtags == 0:
            insights.append("#️⃣ Add 2-3 relevant hashtags for better discoverability")
        elif 2 <= num_hashtags <= 5:
            insights.append("✅ Optimal hashtag count (2-5)")
        elif num_hashtags > 5:
            insights.append("📊 Too many hashtags can reduce engagement")
        
        if '?' in text:
            insights.append("💬 Question detected - encourages comments")
        
        best_hour = 12
        if 'hour' in self.df.columns and 'engagement_rate' in self.df.columns:
            best_hour = int(self.df.groupby('hour')['engagement_rate'].mean().idxmax())
        current_hour = datetime.now().hour
        current_is_best = (current_hour == best_hour)
        
        return {
            'viral_probability': float(probability),
            'sentiment_score': float(sentiment),
            'sentiment_label': 'positive' if sentiment > 0.05 else 'negative' if sentiment < -0.05 else 'neutral',
            'recommendation': self._get_viral_recommendation(probability),
            'insights': insights,
            'best_time_to_post': f"{best_hour}:00",
            'is_best_time': current_is_best,
            'content_score': {
                'length_score': min(100, max(0, 100 - abs(text_length - 150) / 150 * 100)),
                'emotion_score': excitement_score * 100,
                'hashtag_score': min(100, num_hashtags * 20)
            }
        }

    def _get_viral_recommendation(self, probability):
        if probability > 0.8:
            return "🚀 Excellent viral potential! Post immediately for maximum reach"
        elif probability > 0.6:
            return "📈 Strong potential! Optimize posting time for best results"
        elif probability > 0.4:
            return "👍 Moderate potential. Add trending hashtags to boost"
        elif probability > 0.2:
            return "📊 Average potential. Consider more engaging content format"
        else:
            return "💡 Low viral probability. Review content strategy and timing"

    def get_competitor_insights(self):
        if 'User' not in self.df.columns and 'Platform' in self.df.columns and 'Country' in self.df.columns:
            self.df['User'] = self.df['Platform'].astype(str) + '_' + self.df['Country'].astype(str)
        
        if 'User' in self.df.columns:
            user_stats = self.df.groupby('User').agg({
                'engagement_rate': 'mean',
                'Likes': 'mean',
                'Retweets': 'mean',
                'vader_compound': 'mean'
            }).round(2)
            top_performers = user_stats.nlargest(5, 'engagement_rate')
            top_performers_dict = top_performers.to_dict('index')
        else:
            top_performers_dict = {}
        
        return {
            'top_performers': top_performers_dict,
            'industry_benchmark': {
                'avg_engagement_rate': self.df['engagement_rate'].mean() if 'engagement_rate' in self.df.columns else 0,
                'top_quartile_engagement': self.df['engagement_rate'].quantile(0.75) if 'engagement_rate' in self.df.columns else 0,
                'sentiment_benchmark': self.df['vader_compound'].mean() if 'vader_compound' in self.df.columns else 0
            }
        }

    def get_content_calendar(self, days=7):
        hour_engagement = self.df.groupby('hour')['engagement_rate'].mean() if 'hour' in self.df.columns else pd.Series([0.5]*24)
        dow_engagement = self.df.groupby('day_of_week')['engagement_rate'].mean() if 'day_of_week' in self.df.columns else pd.Series([0.5]*7)
        
        best_hour = int(hour_engagement.idxmax()) if not hour_engagement.empty else 12
        
        calendar = []
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            dow = date.weekday()
            calendar.append({
                'date': date.strftime('%Y-%m-%d'),
                'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][dow],
                'optimal_time': f"{best_hour}:00",
                'expected_engagement': float(dow_engagement[dow] * 100) if dow < len(dow_engagement) else 50.0,
                'content_type': self._suggest_content_type(dow)
            })
        return calendar
    
    def _suggest_content_type(self, day):
        if day < 4:
            return "Educational/Informative"
        elif day == 4:
            return "Entertainment/Fun"
        else:
            return "Lifestyle/Inspirational"
