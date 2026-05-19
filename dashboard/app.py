import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import numpy as np

# Get the absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add the project root to Python path
sys.path.insert(0, project_root)

from src.trend_analyzer_new import TrendAnalyzerNew as TrendAnalyzer

# Page configuration
st.set_page_config(
    page_title="Social Media Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("📊 Social Media Intelligence Platform")
    st.markdown("<center><i>AI-Powered Analytics for Data-Driven Social Media Strategy</i></center>", unsafe_allow_html=True)
st.markdown("---")

# Load analyzer
@st.cache_resource(ttl=3600)
def load_analyzer():
    data_file = os.path.join(project_root, 'data', 'processed_data.csv')
    if not os.path.exists(data_file):
        st.error(f"❌ Data file not found at: {data_file}")
        return None
    return TrendAnalyzer(data_file)

analyzer = load_analyzer()

if analyzer is None:
    st.stop()

# Sidebar with filters
st.sidebar.markdown("# 🎛️ Filters & Controls")
st.sidebar.markdown("---")

# Date range filter
if 'Timestamp' in analyzer.df.columns:
    min_date = analyzer.df['Timestamp'].min()
    max_date = analyzer.df['Timestamp'].max()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply filter
    if len(date_range) == 2:
        mask = (analyzer.df['Timestamp'].dt.date >= date_range[0]) & \
               (analyzer.df['Timestamp'].dt.date <= date_range[1])
        filtered_df = analyzer.df[mask]
    else:
        filtered_df = analyzer.df
else:
    filtered_df = analyzer.df

# Platform filter
if 'Platform' in filtered_df.columns:
    platforms = ['All'] + list(filtered_df['Platform'].unique())
    selected_platform = st.sidebar.selectbox("📱 Platform", platforms)
    if selected_platform != 'All':
        filtered_df = filtered_df[filtered_df['Platform'] == selected_platform]

# Sentiment filter
sentiment_options = ['All', 'positive', 'neutral', 'negative']
selected_sentiment = st.sidebar.selectbox("😊 Sentiment", sentiment_options)
if selected_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['vader_sentiment'] == selected_sentiment]

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro Tip:** Use filters to analyze specific segments of your audience")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Executive Dashboard", 
    "#️⃣ Hashtag Intelligence", 
    "💬 Sentiment Analytics", 
    "🎯 Viral Predictor",
    "📅 Content Calendar"
])

# Tab 1: Executive Dashboard
with tab1:
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📊 Total Posts", 
            len(filtered_df),
            delta=f"{len(filtered_df) - len(analyzer.df)} from total"
        )
    with col2:
        st.metric(
            "❤️ Avg Likes", 
            f"{filtered_df['Likes'].mean():.0f}",
            delta=f"{((filtered_df['Likes'].mean() - analyzer.df['Likes'].mean()) / analyzer.df['Likes'].mean() * 100):.1f}%"
        )
    with col3:
        st.metric(
            "🔄 Avg Retweets", 
            f"{filtered_df['Retweets'].mean():.0f}",
            delta=f"{((filtered_df['Retweets'].mean() - analyzer.df['Retweets'].mean()) / analyzer.df['Retweets'].mean() * 100):.1f}%"
        )
    with col4:
        engagement_rate = filtered_df['engagement_rate'].mean() * 100
        st.metric(
            "📈 Engagement Rate", 
            f"{engagement_rate:.1f}%",
            delta=f"{engagement_rate - analyzer.df['engagement_rate'].mean() * 100:.1f}%"
        )
    with col5:
        top_sentiment = filtered_df['vader_sentiment'].mode()[0]
        st.metric(
            "😊 Dominant Sentiment", 
            top_sentiment.upper(),
            delta="Positive" if top_sentiment == 'positive' else "Monitor"
        )
    
    st.markdown("---")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Engagement by Hour")
        hour_engagement = filtered_df.groupby('hour')['engagement_rate'].mean().reset_index()
        fig = px.line(hour_engagement, x='hour', y='engagement_rate', 
                      markers=True, title="When to Post for Maximum Engagement")
        fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Engagement Rate")
        fig.add_hline(y=hour_engagement['engagement_rate'].mean(), 
                      line_dash="dash", line_color="red", 
                      annotation_text="Average")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Platform Performance")
        if 'Platform' in filtered_df.columns:
            platform_perf = filtered_df.groupby('Platform').agg({
                'engagement_rate': 'mean',
                'Likes': 'mean'
            }).reset_index()
            fig = px.bar(platform_perf, x='Platform', y='engagement_rate', 
                        color='Likes', title="Engagement by Platform")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Platform data not available in current dataset")
    
    # Recommendations section
    st.markdown("---")
    st.subheader("💡 AI-Powered Recommendations")
    
    analytics = analyzer.get_engagement_analytics()
    recommendations = analytics.get('recommendations', [])
    
    cols = st.columns(len(recommendations))
    for idx, rec in enumerate(recommendations):
        with cols[idx]:
            st.markdown(f"""
            <div class="insight-box">
                <h3>{rec['icon']} {rec['title']}</h3>
                <p>{rec['message']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Top content section
    st.markdown("---")
    st.subheader("🏆 Top Performing Content")
    
    top_content = filtered_df.nlargest(5, 'engagement_rate')[
        ['Text', 'engagement_rate', 'Likes', 'Retweets', 'vader_sentiment']
    ]
    
    for idx, row in top_content.iterrows():
        sentiment_icon = "😊" if row['vader_sentiment'] == 'positive' else "😐" if row['vader_sentiment'] == 'neutral' else "😞"
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <b>{sentiment_icon} {row['Text'][:100]}...</b><br>
            📊 Engagement Rate: {row['engagement_rate']*100:.1f}% | ❤️ {row['Likes']:.0f} likes | 🔄 {row['Retweets']:.0f} retweets
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Hashtag Intelligence
with tab2:
    st.header("#️⃣ Hashtag Intelligence Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        trending_hashtags = analyzer.analyze_hashtag_trends(20)
        
        # Create advanced visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=trending_hashtags['trend_probability'][:10],
            y=trending_hashtags['hashtag'][:10],
            orientation='h',
            marker=dict(
                color=trending_hashtags['trend_probability'][:10],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Trend Probability")
            ),
            text=trending_hashtags['trend_probability'][:10].apply(lambda x: f'{x*100:.1f}%'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top Trending Hashtags by Probability",
            xaxis_title="Trend Probability",
            yaxis_title="Hashtag",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Hashtag Metrics")
        
        # Hashtag stats
        avg_hashtags = filtered_df['num_hashtags'].mean()
        st.metric("Average Hashtags per Post", f"{avg_hashtags:.1f}")
        
        best_count = filtered_df.groupby('num_hashtags')['engagement_rate'].mean().idxmax()
        st.metric("Optimal Hashtag Count", f"{best_count:.0f}")
        
        st.markdown("---")
        st.subheader("🎯 Top 5 Recommendations")
        
        for idx, row in trending_hashtags.head(5).iterrows():
            confidence = row.get('confidence', 70)
            growth = row.get('growth_trend', '📈 Rising')
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #4CAF50 {confidence}%, #e0e0e0 {confidence}%); 
                        padding: 10px; border-radius: 5px; margin: 5px 0;">
                <b>#{row['hashtag']}</b><br>
                <small>🎯 {row['trend_probability']*100:.1f}% trend probability | {growth}</small>
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Sentiment Analytics
with tab3:
    st.header("💬 Advanced Sentiment Analytics")
    
    sentiment_trends = analyzer.get_sentiment_trends(60)
    
    # Create subplot for sentiment analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Over Time', 'Sentiment Distribution',
                       'Sentiment by Hour', 'Anomaly Detection'),
        specs=[[{"secondary_y": True}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Sentiment over time
    fig.add_trace(
        go.Scatter(x=sentiment_trends['date'], y=sentiment_trends['sentiment_ma7'],
                  name='7-Day MA', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sentiment_trends['date'], y=sentiment_trends['trend_rate'],
                  name='Trending Rate', line=dict(color='red', width=2, dash='dot')),
        row=1, col=1, secondary_y=True
    )
    
    # Sentiment distribution
    sentiment_dist = filtered_df['vader_sentiment'].value_counts()
    fig.add_trace(
        go.Pie(labels=sentiment_dist.index, values=sentiment_dist.values,
               marker=dict(colors=['#4CAF50', '#FFC107', '#F44336'])),
        row=1, col=2
    )
    
    # Sentiment by hour
    hour_sentiment = filtered_df.groupby('hour')['vader_compound'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=hour_sentiment['hour'], y=hour_sentiment['vader_compound'],
                  mode='lines+markers', fill='tozeroy'),
        row=2, col=1
    )
    
    # Anomaly detection
    print(sentiment_trends.columns)
    if 'is_anomaly' in sentiment_trends.columns:
        anomalies = sentiment_trends[
            sentiment_trends['is_anomaly'] == True
        ]
    else:
        anomalies = pd.DataFrame()
    if not anomalies.empty and 'date' in anomalies.columns:

        fig.add_trace(
            go.Bar(
                x=anomalies['date'],
                y=anomalies['sentiment_zscore'],
                name='Sentiment Anomalies',
                marker_color='orange'
            ),
            row=2,
            col=2
        )
    
    fig.update_layout(height=700, showlegend=True, title_text="Sentiment Intelligence Dashboard")
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_yaxes(title_text="Trending Rate", row=1, col=1, secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_sentiment = filtered_df['vader_compound'].mean()
        st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}",
                 delta="Positive" if avg_sentiment > 0 else "Negative")
    
    with col2:
        sentiment_volatility = filtered_df['vader_compound'].std()
        st.metric("Sentiment Volatility", f"{sentiment_volatility:.3f}",
                 delta="High variation" if sentiment_volatility > 0.3 else "Stable")
    
    with col3:
        positive_ratio = (sentiment_dist.get('positive', 0) / len(filtered_df)) * 100
        st.metric("Positive Content Ratio", f"{positive_ratio:.1f}%")

# Tab 4: Viral Predictor
with tab4:
    st.header("🎯 AI Viral Potential Predictor")
    st.markdown("Enter your content below to get a detailed viral prediction analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        post_text = st.text_area(
            "📝 Post Content",
            height=150,
            placeholder="Write your post content here...",
            help="Longer, engaging content typically performs better"
        )
        
        hashtags = st.text_input(
            "#️⃣ Hashtags",
            placeholder="#example #socialmedia #viral",
            help="Use 3-5 relevant hashtags for optimal reach"
        )
        
        # Optional fields
        use_current_time = st.checkbox("Use current time for prediction", value=True)
        
        if st.button("🔮 Predict Viral Potential", type="primary", use_container_width=True):
            if post_text:
                with st.spinner("Analyzing content with AI..."):
                    prediction = analyzer.predict_viral_potential(post_text, hashtags)
                    
                    # Results display
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    # Viral probability gauge
                    prob = prediction['viral_probability']
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prob * 100,
                        title = {'text': "Viral Potential Score"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgray"},
                                {'range': [33, 66], 'color': "gray"},
                                {'range': [66, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment Score", f"{prediction['sentiment_score']:.3f}",
                                 delta=prediction['sentiment_label'])
                    with col2:
                        st.metric("Best Time to Post", prediction['best_time_to_post'],
                                 delta="Optimal" if prediction['is_best_time'] else "Adjust timing")
                    with col3:
                        content_score = np.mean(list(prediction['content_score'].values()))
                        st.metric("Content Quality Score", f"{content_score:.0f}/100")
                    
                    # Insights
                    st.markdown("---")
                    st.subheader("💡 AI Insights & Recommendations")
                    
                    st.info(prediction['recommendation'])
                    
                    if prediction['insights']:
                        st.markdown("**Key Insights:**")
                        for insight in prediction['insights']:
                            st.markdown(f"- {insight}")
                    
                    # Content score breakdown
                    st.markdown("---")
                    st.subheader("📈 Content Score Breakdown")
                    
                    score_df = pd.DataFrame([
                        {"Metric": "Length", "Score": prediction['content_score']['length_score']},
                        {"Metric": "Emotion", "Score": prediction['content_score']['emotion_score']},
                        {"Metric": "Hashtags", "Score": prediction['content_score']['hashtag_score']}
                    ])
                    
                    fig = px.bar(score_df, x='Metric', y='Score', 
                                color='Metric', range_y=[0, 100],
                                title="Content Optimization Metrics")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please enter post content to analyze")
    
    with col2:
        st.markdown("""
        ### 📋 Best Practices for Viral Content
        
        **✅ DO's:**
        - Use engaging visuals (images/videos)
        - Post during peak hours (9-11 AM or 7-9 PM)
        - Include 3-5 relevant hashtags
        - Create emotional resonance
        - Encourage engagement (questions, polls)
        
        **❌ DON'Ts:**
        - Overuse hashtags (>5)
        - Post low-quality content
        - Ignore audience sentiment
        - Copy content without value
        - Post during off-peak hours
        
        ---
        
        ### 🎯 Trending Topics
        """)
        
        # Show trending hashtags
        trending = analyzer.analyze_hashtag_trends(5)
        for idx, row in trending.iterrows():
            st.markdown(f"- **#{row['hashtag']}** - {row['trend_probability']*100:.0f}% trend probability")

# Tab 5: Content Calendar
with tab5:
    st.header("📅 Smart Content Calendar")
    st.markdown("AI-generated optimal posting schedule for maximum engagement")
    
    calendar = analyzer.get_content_calendar(7)
    
    # Create calendar display
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Engagement by day heatmap
    dow_engagement = filtered_df.groupby('day_of_week')['engagement_rate'].mean().reset_index()
    dow_engagement['day'] = dow_engagement['day_of_week'].apply(lambda x: days[x])
    
    fig = px.bar(dow_engagement, x='day', y='engagement_rate', 
                 color='engagement_rate', color_continuous_scale='Viridis',
                 title="Expected Engagement by Day of Week")
    st.plotly_chart(fig, use_container_width=True)
    
    # Calendar table
    st.markdown("---")
    st.subheader("📆 Weekly Posting Schedule")
    
    calendar_df = pd.DataFrame(calendar)
    calendar_df['expected_engagement'] = calendar_df['expected_engagement'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        calendar_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": "Date",
            "day": "Day",
            "optimal_time": "Best Time",
            "expected_engagement": st.column_config.ProgressColumn(
                "Expected Engagement",
                help="Expected engagement rate for this day",
                format="%",
                min_value=0,
                max_value=100
            ),
            "content_type": "Suggested Content"
        }
    )
    
    # Pro tips
    st.markdown("---")
    st.info("💡 **Pro Tip:** Schedule your most important content for days with highest expected engagement rates. Use the optimal times for each day to maximize reach.")

# Footer
st.markdown("---")
st.markdown("""
<center>
    <small>🚀 Powered by AI & Machine Learning | Real-time Social Media Intelligence | Updated Daily</small>
</center>
""", unsafe_allow_html=True)