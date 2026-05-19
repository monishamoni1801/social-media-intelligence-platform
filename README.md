# 📊 Social Media Intelligence Platform

An AI-powered Social Media Analytics and Viral Trend Prediction Platform built using Machine Learning, NLP, and Streamlit.

This project analyzes social media content, predicts viral potential, performs sentiment analysis, identifies trending hashtags, and generates smart content recommendations.

---

# 🚀 Features

## ✅ Executive Analytics Dashboard
- Total posts analytics
- Average likes and retweets
- Engagement rate tracking
- Platform performance insights
- AI-powered recommendations

---

## ✅ Hashtag Intelligence System
- Trending hashtag detection
- Hashtag performance analysis
- Trend probability prediction
- Growth trend analysis
- Smart hashtag recommendations

---

## ✅ Advanced Sentiment Analytics
- VADER sentiment analysis
- Sentiment trend tracking
- Sentiment anomaly detection
- Positive/negative/neutral classification
- Sentiment visualization dashboard

---

## ✅ Viral Content Predictor
- AI-based viral probability prediction
- Content quality analysis
- Best posting time recommendation
- Emotional content analysis
- Engagement optimization suggestions

---

## ✅ Smart Content Calendar
- AI-generated posting schedule
- Optimal posting time prediction
- Expected engagement estimation
- Content strategy planning

---

# 🧠 Technologies Used

## Frontend
- Streamlit
- Plotly

## Backend
- Python

## Machine Learning / NLP
- Scikit-learn
- XGBoost
- NLTK
- TextBlob

## Data Processing
- Pandas
- NumPy

---

# 📂 Project Structure

```bash
social-media-intelligence-platform/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── analysis_engine.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── trend_analyzer_new.py
│
├── data/
│   ├── social_media_data.csv
│   └── processed_data.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── model graphs
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

# ⚙️ Installation Guide

## Step 1 — Clone Repository

```bash
git clone https://github.com/monishamoni1801/social-media-intelligence-platform.git
```

---

## Step 2 — Move Into Project Folder

```bash
cd social-media-intelligence-platform
```

---

## Step 3 — Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5 — Download NLTK Packages

Run Python:

```bash
python
```

Then run:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
```

Exit:

```python
exit()
```

---

# ▶️ Run the Application

```bash
streamlit run dashboard/app.py
```

---

# 📊 Machine Learning Workflow

## 1️⃣ Data Collection
- Social media posts dataset
- Likes, retweets, hashtags, timestamps

---

## 2️⃣ Feature Engineering
Generated features:
- Hashtag count
- Average hashtag length
- Engagement rate
- Like-retweet ratio
- Sentiment scores
- Temporal features

---

## 3️⃣ NLP Processing
- Text cleaning
- Tokenization
- Lemmatization
- Sentiment analysis

---

## 4️⃣ Model Training
Models used:
- Random Forest
- XGBoost

---

## 5️⃣ Prediction System
Predicts:
- Viral probability
- Trending hashtags
- Audience sentiment
- Best posting time

---

# 📈 Dashboard Modules

| Module | Description |
|---|---|
| Executive Dashboard | Business analytics overview |
| Hashtag Intelligence | Trending hashtag analysis |
| Sentiment Analytics | Audience sentiment tracking |
| Viral Predictor | Viral probability prediction |
| Content Calendar | AI posting schedule |

---

# 🧪 Example Features

## Viral Prediction Input
- Post text
- Hashtags
- Posting time

## Viral Prediction Output
- Viral probability score
- Sentiment label
- Best posting time
- AI recommendations

---

# 📸 Screenshots

Add screenshots here later.

Example:

```md
![Dashboard](screenshots/dashboard.png)
```

---

# 🔥 Future Improvements

- Real-time Twitter/X API integration
- Instagram analytics support
- Deep learning models
- AI-generated captions
- Real-time trend monitoring
- User authentication
- Cloud deployment

---

# 👩‍💻 Author

## Monisha

AI/ML & Data Science Enthusiast

---

# 📄 License

This project is for educational and research purposes.

---

# ⭐ If You Like This Project

Give this repository a star ⭐
