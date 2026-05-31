import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sentiment Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .block-container { padding: 2rem 3rem; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin: 0; letter-spacing: 0.05em; text-transform: uppercase; }
    .pos { color: #3fb950; }
    .neg { color: #f85149; }
    .neu { color: #8b949e; }
    .avg { color: #58a6ff; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stDataFrame { background: #161b22; }
    div[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem; }
    .section-title { font-size: 1rem; font-weight: 600; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# fetch and process data
@st.cache_data(ttl=300)
def fetch_data():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", headers=headers).json()[:100]
    data = []
    for story_id in ids:
        story = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", headers=headers).json()
        if story and "title" in story:
            data.append({
                "title": story.get("title", ""),
                "score": story.get("score", 0),
                "comments": story.get("descendants", 0),
                "created_utc": story.get("time", 0)
            })
    return pd.DataFrame(data)

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze(df):
    analyzer = SentimentIntensityAnalyzer()
    df["clean_title"] = df["title"].apply(clean)
    df["compound"] = df["clean_title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment"] = df["compound"].apply(
        lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral"))
    df["date"] = pd.to_datetime(df["created_utc"], unit="s")
    return df

# header
st.markdown("# 📡 HackerNews Sentiment Radar")
st.markdown("<p style='color:#8b949e; margin-top:-1rem;'>Real-time NLP analysis of top tech stories</p>", unsafe_allow_html=True)

# refresh button
col_r, _ = st.columns([1, 5])
with col_r:
    if st.button("⟳ Refresh Data"):
        st.cache_data.clear()

# load data
with st.spinner("Fetching live data from HackerNews..."):
    df = analyze(fetch_data())

st.markdown("---")

# metric cards
st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='metric-card'><p class='metric-value neu'>{len(df)}</p><p class='metric-label'>Total Posts</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><p class='metric-value pos'>{len(df[df.sentiment=='positive'])}</p><p class='metric-label'>Positive</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><p class='metric-value neg'>{len(df[df.sentiment=='negative'])}</p><p class='metric-label'>Negative</p></div>", unsafe_allow_html=True)
with c4:
    avg = round(df["compound"].mean(), 3)
    st.markdown(f"<div class='metric-card'><p class='metric-value avg'>{avg}</p><p class='metric-label'>Avg Sentiment</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# charts row
st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # pie chart
    fig_pie = px.pie(
        df, names="sentiment",
        color="sentiment",
        color_discrete_map={"positive": "#3fb950", "negative": "#f85149", "neutral": "#8b949e"},
        hole=0.5,
        title="Sentiment Distribution"
    )
    fig_pie.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#c9d1d9", title_font_color="#e6edf3",
        legend=dict(bgcolor="#161b22")
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # bar chart top scoring posts
    top10 = df.nlargest(10, "score")
    fig_bar = px.bar(
        top10, x="score", y="title",
        orientation="h",
        color="sentiment",
        color_discrete_map={"positive": "#3fb950", "negative": "#f85149", "neutral": "#58a6ff"},
        title="Top 10 Posts by Score"
    )
    fig_bar.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#c9d1d9", title_font_color="#e6edf3",
        yaxis=dict(tickfont=dict(size=9)),
        showlegend=False,
        margin=dict(l=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# word cloud
st.markdown("<div class='section-title'>Word Cloud</div>", unsafe_allow_html=True)
all_text = " ".join(df["clean_title"].tolist())
stopwords = {"the","a","an","is","in","of","to","and","for","on","with","by","from","that","this","it","at","as","be","are","was","has","have","will","its","or","not","but","about","how","what","show","hn","ask","tell"}
wc = WordCloud(
    width=1200, height=300,
    background_color="#0d1117",
    colormap="Blues",
    stopwords=stopwords,
    max_words=80
).generate(all_text)

fig_wc, ax = plt.subplots(figsize=(14, 3))
fig_wc.patch.set_facecolor("#0d1117")
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)

st.markdown("<br>", unsafe_allow_html=True)

# search and filter
st.markdown("<div class='section-title'>Explore Posts</div>", unsafe_allow_html=True)
col_s, col_f = st.columns([3, 1])
with col_s:
    search = st.text_input("", placeholder="🔍 Search posts by keyword...")
with col_f:
    sentiment_filter = st.selectbox("Filter by sentiment", ["all", "positive", "negative", "neutral"])

filtered = df.copy()
if search:
    filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]
if sentiment_filter != "all":
    filtered = filtered[filtered["sentiment"] == sentiment_filter]

st.dataframe(
    filtered[["title", "sentiment", "compound", "score", "comments"]].sort_values("compound"),
    use_container_width=True,
    height=400
)

st.markdown(f"<p style='color:#8b949e; font-size:0.8rem;'>Showing {len(filtered)} of {len(df)} posts</p>", unsafe_allow_html=True)