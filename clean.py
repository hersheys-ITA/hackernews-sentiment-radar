import re
import pandas as pd

df = pd.read_csv("posts.csv")
print("Loaded", len(df), "posts")

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_title"] = df["title"].apply(clean)
df.to_csv("posts_clean.csv", index=False)

print("Cleaned! Sample:")
print(df[["title", "clean_title"]].head(5))