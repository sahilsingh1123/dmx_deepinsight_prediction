pyinstaller --additional-hooks-dir=. di_ml_server.py entry.py FPGrowth.py KMeans.py SentimentAnalysis.py Forecasting.py
cp -R packed_nltk_data dist/di_ml_server/
