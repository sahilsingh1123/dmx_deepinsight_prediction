import nltk
import time

nltk.data.path.append(r'packed_nltk_data')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def perform_sentiment_analysis(data):
    print('new sentiment analysis request!')
    start_time = time.time()
    scores = []
    sid = SentimentIntensityAnalyzer()
    for text in data:
        ss = sid.polarity_scores(text)
        neg = ss['neg']
        neu = ss['neu']
        pos = ss['pos']
        compound = (ss['compound'] + 1.0) / 2.0
        scores.append([neg, neu, pos, compound])
    print('sentiment analysis complete for ' + str(len(data)) + ' texts!')
    json_response = {'run_status': 'success', 'sentimentScores': scores, 'execution_time': time.time() - start_time}
    #return str(json.dumps(json_response)).encode('utf-8')
    return json_response
