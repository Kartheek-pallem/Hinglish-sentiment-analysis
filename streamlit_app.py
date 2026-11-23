"""WhatsApp Hinglish sentiment dashboard (the base paper's Streamlit app, with our models).

    streamlit run app/streamlit_app.py
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from predict import Predictor, available_models   # noqa: E402
from preprocess import SAFE_STOPWORDS, clean        # noqa: E402
from whatsapp_parser import parse_chat              # noqa: E402

COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
ORDER = ["negative", "neutral", "positive"]

st.set_page_config(page_title="Hinglish WhatsApp Sentiment", layout="wide")
st.title("Hinglish WhatsApp Sentiment Analyzer")
st.caption("CS 725 project · models trained on SentiMix (SemEval-2020 Task 9)")


@st.cache_resource
def get_model(name):
    return Predictor(name)


models = available_models()
if not models:
    st.error("No trained model found. Run `python src/classical.py` first (takes ~2 minutes).")
    st.stop()

with st.sidebar:
    model_name = st.selectbox("Model", models, help="svm is fastest; hingbert is most accurate")
    dayfirst = st.checkbox("Dates are day/month (India)", value=True)
    upload = st.file_uploader("WhatsApp chat export (.txt)", type="txt")
    use_sample = st.button("Use sample chat")

predictor = get_model(model_name)

with st.expander("Try one message"):
    msg = st.text_input("Message", "match dekh ke maza aa gaya yaar 🔥")
    if msg:
        label, prob = predictor.predict([msg])
        st.write(f"**{label[0]}**  ·  " + "  ".join(f"{k}: {v:.2f}" for k, v in zip(ORDER, prob[0])))

if use_sample:
    st.session_state["chat"] = (ROOT / "app" / "sample_chat.txt").read_text(encoding="utf-8")
if upload is not None:
    st.session_state["chat"] = upload.read().decode("utf-8", errors="replace")
if "chat" not in st.session_state:
    st.info("Upload a chat (WhatsApp > chat > ⋮ > More > Export chat > Without media) or click 'Use sample chat'.")
    st.stop()


@st.cache_data
def analyse(text, dayfirst, model_name):
    df = parse_chat(text, dayfirst=dayfirst)
    if len(df):
        df["sentiment"], probs = get_model(model_name).predict(df.message.tolist())
        df["confidence"] = probs.max(1)
    return df


df = analyse(st.session_state["chat"], dayfirst, model_name)
if df.empty:
    st.error("Could not read any messages. Is this a WhatsApp export .txt file?")
    st.stop()

users = ["Overall"] + sorted(df.user.unique())
who = st.selectbox("Show analysis for", users)
view = df if who == "Overall" else df[df.user == who]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Messages", len(view))
for col, lab in zip((c2, c3, c4), ORDER[::-1]):
    col.metric(lab.title(), f"{(view.sentiment == lab).mean():.0%}")

# distribution + monthly activity map
left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    counts = view.sentiment.value_counts().reindex(ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
    ax.set_ylabel("messages")
    st.pyplot(fig)
with right:
    st.subheader("Monthly activity map")
    monthly = pd.crosstab(view.month, view.sentiment).reindex(columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    monthly.plot(kind="bar", stacked=True, color=[COLORS[c] for c in ORDER], ax=ax)
    ax.set_xlabel("")
    st.pyplot(fig)

st.subheader("Timeline")
freq = st.radio("Group by", ["Day", "Month"], horizontal=True)
key = "day" if freq == "Day" else "month"
timeline = pd.crosstab(view[key], view.sentiment).reindex(columns=ORDER, fill_value=0)
st.line_chart(timeline, color=[COLORS[c] for c in ORDER])

st.subheader("Weekday activity")
week = pd.crosstab(view.weekday, view.sentiment).reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0)
st.bar_chart(week.reindex(columns=ORDER, fill_value=0), color=[COLORS[c] for c in ORDER])

if who == "Overall":
    st.subheader("Contribution by user")
    contrib = pd.crosstab(df.user, df.sentiment).reindex(columns=ORDER, fill_value=0)
    share = (contrib / contrib.sum() * 100).round(1)
    cols = st.columns(3)
    for col, lab in zip(cols, ["positive", "negative", "neutral"]):
        col.write(f"**Most {lab}** (% of all {lab} messages)")
        col.dataframe(share[lab].sort_values(ascending=False).rename("%"))

st.subheader("Word clouds and most common words")
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None
cols = st.columns(3)
for col, lab in zip(cols, ORDER):
    words = [w for m in view[view.sentiment == lab].message for w in clean(m).split()
             if w.isalpha() and len(w) > 2 and w not in SAFE_STOPWORDS]
    col.write(f"**{lab.title()}**")
    if not words:
        col.write("no messages")
        continue
    if WordCloud:
        wc = WordCloud(width=400, height=250, background_color="white", colormap="viridis").generate(" ".join(words))
        col.image(wc.to_array())
    col.dataframe(pd.DataFrame(Counter(words).most_common(10), columns=["word", "count"]),
                  hide_index=True)

st.subheader("Messages")
st.dataframe(view[["datetime", "user", "message", "sentiment", "confidence"]])
st.download_button("Download results (CSV)", view.to_csv(index=False), "chat_sentiment.csv")
