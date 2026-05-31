import requests
import pandas as pd

print("Fetching posts from HackerNews...")

# get top 100 story IDs
ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:100]

data = []
for i, story_id in enumerate(ids):
    story = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json").json()
    if story and "title" in story:
        data.append({
            "title": story.get("title", ""),
            "score": story.get("score", 0),
            "comments": story.get("descendants", 0),
            "created_utc": story.get("time", 0)
        })
    if i % 10 == 0:
        print(f"Fetched {i}/100...")

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["created_utc"], unit="s")
print(df.head(10))
df.to_csv("posts.csv", index=False)
print("Done! Saved to posts.csv")