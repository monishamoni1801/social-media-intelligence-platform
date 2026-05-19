@echo off
cd /d D:\Monisha\socialmm_project\hashtag_prediction\social-trend-analyzer

echo Checking if processed_data.csv exists...
if exist data\processed_data.csv (
    echo ✅ Data file found!
) else (
    echo ❌ Data file not found! Running preprocessing...
    cd src
    python data_preprocessing.py
    cd ..
)

echo.
echo Launching Streamlit Dashboard...
streamlit run dashboard\app.py

pause