
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
import os
warnings.filterwarnings('ignore')

class TrendAnalyzerNew:
    def __init__(self, data_path='../data/processed_data.csv'):
        if not os.path.exists(data_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_data.csv')
            if os.path.exists(alt_path):
                data_path = alt_path
        self.df = pd.read_csv(data_path)
        self.sia = SentimentIntensityAnalyzer()
        if 'Timestamp' in self.df.columns:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])

    def analyze_hashtag_trends(self, top_n=20, time_window_days=30):
        recent_df = self.df[self.df['Timestamp'] >= self.df['Timestamp'].max() - timedelta(days=time_window_days)]
        hashtags_list = []
        for idx, row in recent_df.iterrows():
            if pd.notna(row.get('Hashtags', '')):
                for ht in str(row['Hashtags']).split():
                    hashtags_list.append({
                        'hashtag': ht.lower().replace('#', ''), 
                        'engagement': row.get('engagement_rate', 0),
                        'sentiment': row.get('vader_compound', 0),
                        'will_trend': row.get('will_trend', 0)
                    })
        if not hashtags_list:
            return pd.DataFrame()
        result = pd.DataFrame(hashtags_list)
        result = result.groupby('hashtag').agg({
            'engagement': 'mean',
            'sentiment': 'mean',
            'will_trend': 'sum'
        }).reset_index()
        result.columns = ['hashtag', 'total_engagement', 'avg_sentiment', 'will_trend']
        result['trend_probability'] = result['total_engagement'] / result['total_engagement'].max() if result['total_engagement'].max() > 0 else 0
        result['growth_trend'] = '📈 Rising'
        result['confidence'] = 75
        result['frequency'] = 1
        return result.sort_values('trend_probability', ascending=False).head(top_n)

    def get_sentiment_trends(self, days=30):

        # Check Timestamp exists
        if 'Timestamp' not in self.df.columns:
            return pd.DataFrame()

        # Create date column
        self.df['date'] = pd.to_datetime(self.df['Timestamp']).dt.date

        # Group daily statistics
        daily = self.df.groupby('date').agg({
            'vader_compound': 'mean',
            'will_trend': 'mean',
            'engagement_rate': 'mean'
        }).reset_index()

        # Rename columns
        daily.columns = [
            'date',
            'sentiment_mean',
            'trend_rate',
            'engagement_rate'
        ]

        # Moving averages
        daily['sentiment_ma7'] = (
            daily['sentiment_mean']
            .rolling(window=7, min_periods=1)
            .mean()
        )

        daily['sentiment_ma30'] = (
            daily['sentiment_mean']
            .rolling(window=30, min_periods=1)
            .mean()
        )

        daily['sentiment_trend'] = (
            daily['sentiment_ma7']
            - daily['sentiment_ma30']
        )

        daily['trend_indicator'] = (
            daily['trend_rate']
            .rolling(window=3, min_periods=1)
            .mean()
        )

        # Anomaly detection
        rolling_mean = (
            daily['sentiment_mean']
            .rolling(window=30, min_periods=1)
            .mean()
        )

        rolling_std = (
            daily['sentiment_mean']
            .rolling(window=30, min_periods=1)
            .std()
        )

        # Avoid divide by zero
        rolling_std = rolling_std.replace(0, 1)

        daily['sentiment_zscore'] = (
            (daily['sentiment_mean'] - rolling_mean)
            / rolling_std
        )

        # Detect anomalies
        daily['is_anomaly'] = (
            daily['sentiment_zscore'].abs() > 2
        )

        # Fill missing values
        daily['is_anomaly'] = (
            daily['is_anomaly']
            .fillna(False)
            .astype(bool)
        )

        daily['sentiment_zscore'] = (
            daily['sentiment_zscore']
            .fillna(0)
        )

        # Trend direction
        daily['trend_direction'] = '➡️ Stable'

        daily.loc[
            daily['sentiment_trend'] > 0.1,
            'trend_direction'
        ] = '📈 Improving'

        daily.loc[
            daily['sentiment_trend'] < -0.1,
            'trend_direction'
        ] = '📉 Declining'

        # Extra dashboard columns
        daily['post_count'] = 1

        daily['sentiment_std'] = (
            daily['sentiment_mean']
            .rolling(window=7, min_periods=1)
            .std()
            .fillna(0)
        )

        return daily.tail(days)

    def get_engagement_analytics(self):
        best_hour = int(self.df.groupby('hour')['engagement_rate'].mean().idxmax()) if 'hour' in self.df.columns else 12
        return {
            'avg_likes': float(self.df['Likes'].mean()),
            'avg_retweets': float(self.df['Retweets'].mean()),
            'total_posts': len(self.df),
            'unique_hashtags': int(self.df['Hashtags'].nunique()) if 'Hashtags' in self.df.columns else 0,
            'engagement_by_hour': self.df.groupby('hour')['engagement_rate'].mean().to_dict(),
            'engagement_std_by_hour': self.df.groupby('hour')['engagement_rate'].std().to_dict(),
            'best_time_to_post': best_hour,
            'worst_time_to_post': int(self.df.groupby('hour')['engagement_rate'].mean().idxmin()) if 'hour' in self.df.columns else 0,
            'best_day': 'Monday',
            'platform_stats': {},
            'sentiment_distribution': self.df['vader_sentiment'].value_counts().to_dict(),
            'dominant_sentiment': str(self.df['vader_sentiment'].mode()[0]) if len(self.df['vader_sentiment'].mode()) > 0 else 'neutral',
            'top_performing_content': self.df.nlargest(10, 'engagement_rate')[['Text', 'engagement_rate', 'Likes', 'Retweets', 'vader_sentiment', 'Hashtags']].to_dict('records'),
            'engagement_trend': {'weekly_growth': 0, 'likes_growth': 0, 'retweets_growth': 0},
            'recommendations': [
                {'type': 'timing', 'title': 'Optimal Time', 'message': f'Best: {best_hour}:00', 'icon': '⏰'},
                {'type': 'content', 'title': 'Content Strategy', 'message': f'{self.df["vader_sentiment"].mode()[0].capitalize() if len(self.df["vader_sentiment"].mode()) > 0 else "Positive"} posts work best', 'icon': '💡'}
            ]
        }

    def predict_viral_potential(self, text, hashtags):
        sentiment = self.sia.polarity_scores(text)['compound']
        text_len = len(text)
        ht_count = len(hashtags.split()) if hashtags else 0
        
        score = 0
        if 100 <= text_len <= 200:
            score += 30
        elif 50 <= text_len <= 300:
            score += 15
            
        excitement = sum(1 for w in ['amazing','incredible','awesome','excited','love','best','perfect','crazy'] if w in text.lower())
        score += min(25, excitement * 8)
        
        if 2 <= ht_count <= 5:
            score += 20
        elif ht_count > 0:
            score += 10
            
        if any(c in text for c in ['🎉','🔥','🚀','💪','❤️']):
            score += 15
        if '?' in text:
            score += 15
        if text.count('!') >= 2:
            score += 10
        if sentiment > 0.3:
            score += 20
        elif sentiment > 0:
            score += 10
        if any(w in text.lower() for w in ['comment','share','tag','follow']):
            score += 10
            
        probability = min(0.95, max(0.05, score / 100))
        
        insights = []
        if text_len < 50:
            insights.append("📝 Short posts work well on Twitter")
        elif 100 <= text_len <= 200:
            insights.append("✅ Optimal length")
        if ht_count == 0:
            insights.append("#️⃣ Add 2-3 hashtags")
        elif 2 <= ht_count <= 5:
            insights.append("✅ Good hashtag count")
        if excitement >= 2:
            insights.append("⚡ Strong emotional language")
        if '?' in text:
            insights.append("💬 Questions boost comments")
            
        if probability > 0.7:
            rec = "🚀 HIGH VIRAL POTENTIAL! Post now!"
        elif probability > 0.5:
            rec = "📈 Good potential! Optimize timing"
        elif probability > 0.3:
            rec = "👍 Moderate potential"
        else:
            rec = "💡 Low viral probability"
            
        best_hour = int(self.df.groupby('hour')['engagement_rate'].mean().idxmax()) if 'hour' in self.df.columns else 12
        
        return {
            'viral_probability': probability,
            'sentiment_score': sentiment,
            'sentiment_label': 'positive' if sentiment > 0.05 else 'negative' if sentiment < -0.05 else 'neutral',
            'recommendation': rec,
            'insights': insights,
            'best_time_to_post': f"{best_hour}:00",
            'is_best_time': (datetime.now().hour == best_hour),
            'content_score': {
                'length_score': min(100, max(0, 100 - abs(text_len - 150) / 150 * 100)),
                'emotion_score': excitement * 20,
                'hashtag_score': ht_count * 20
            }
        }

    def get_competitor_insights(self):
        return {
            'top_performers': {}, 
            'industry_benchmark': {
                'avg_engagement_rate': float(self.df['engagement_rate'].mean()), 
                'top_quartile_engagement': float(self.df['engagement_rate'].quantile(0.75)),
                'sentiment_benchmark': float(self.df['vader_compound'].mean())
            }
        }

    def get_content_calendar(self, days=7):
        best_hour = int(self.df.groupby('hour')['engagement_rate'].mean().idxmax()) if 'hour' in self.df.columns else 12
        dow_engagement = self.df.groupby('day_of_week')['engagement_rate'].mean() if 'day_of_week' in self.df.columns else pd.Series([0.5]*7)
        calendar = []
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            dow = date.weekday()
            calendar.append({
                'date': date.strftime('%Y-%m-%d'), 
                'day': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][dow], 
                'optimal_time': f"{best_hour}:00", 
                'expected_engagement': float(dow_engagement[dow] * 100) if dow < len(dow_engagement) else 50.0,
                'content_type': "Educational" if dow < 4 else "Entertainment"
            })
        return calendar
