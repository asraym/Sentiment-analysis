import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dataset Sentiment Analyzer", layout="centered")
st.title("Sentiment Analysis of Feedback Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
  try:
    df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
    st.subheader("Preview of uploaded Data")
    st.write(df.head())
    column = st.selectbox("Select the column containing feedback", df.columns)
    if st.button("Analyze Sentiment"):
      st.info("Analyzing...")
      def get_sentiment(text):
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0:
          return "Positive"
        elif polarity < 0:
          return "Negative"
        else:
          return "Neutral"
      df['Sentiment'] = df[column].apply(get_sentiment)  
      sentiment_counts = df['Sentiment'].value_counts()
      total = len(df)
      st.subheader("Sentiment Breakdown")
      for sentiment in ['Positive', 'Neutral', 'Negative']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / total) * 100
        st.write(f"{sentiment}: {percentage:.2f}% ({count} out of {total})")
      st.subheader("Pie Chart")
      fig, ax = plt.subplots()
      ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
      ax.axis('equal')
      st.pyplot(fig)
      st.subheader("Detailed Sentiment Analysis")
      st.write(df)

  except Exception as e:
      st.error(f"Error: {e}")
