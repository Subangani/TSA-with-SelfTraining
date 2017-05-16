emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative': -0.5, 'Extremely-Negative': -1.0, 'Neutral': 0.0}

print emo_scores
sentiment = "Positive"
xx=emo_scores.get(sentiment)
print xx