from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

df = pd.read_csv("posts_clean.csv")
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["compound"] = df["clean_title"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment"] = df["clean_title"].apply(get_sentiment)

df.to_csv("posts_sentiment.csv", index=False)

print("Done! Sentiment breakdown:")
print(df["sentiment"].value_counts())
print("\nSample:")
print(df[["title", "sentiment", "compound"]].head(10))