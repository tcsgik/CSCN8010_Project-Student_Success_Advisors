# Import libraries
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

distressed_emotions = ["anger", "sadness", "fear", "disgust"]
# Load log CSV file
df = pd.read_csv('logs/log.csv', parse_dates=['datetime'])
df["is_distressed"] = df['emotion'].isin(distressed_emotions)
st.title("📊 Chatbot Dashboard")

# Intent type distribution map
st.header("✨ FAQ Intent Type Distribution")
intent_counts = df['intent'].value_counts()
# Sort values
intent_counts = intent_counts.sort_values(ascending=False)
# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=intent_counts.values, y=intent_counts.index, ax=ax, palette='Blues_d')
# Labels and title
ax.set_xlabel('Count')
ax.set_ylabel('Intent Type')
ax.set_title('Intent Type Distribution')
# Layout and display
plt.tight_layout()
st.pyplot(fig)

# Percentage of troublesome messages
st.header("⚠️ Percentage of Disturbing Messages")
distress_counts = df['is_distressed'].value_counts().rename({True: 'Distressed', False: 'Not Distressed'})

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(distress_counts.index, distress_counts.values, color=['red', 'green'])
ax2.set_ylabel('Count')
ax2.set_title('Distress Message Proportion')
st.pyplot(fig2)

# Heat map of troublesome messages
st.header("📆 Daily Distress Information Heatmap")
df['date'] = df['datetime'].dt.date
pivot = df.pivot_table(index='date', columns='is_distressed', values='student', aggfunc='count', fill_value=0)
pivot.rename(columns={False: 'Not Distressed', True: 'Distressed'}, inplace=True)

fig3, ax3 = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt="d", cmap='YlOrRd', ax=ax3)
ax3.set_title('Daily Distress Messages Heatmap')
st.pyplot(fig3)

# Latest trouble information list
st.header("🚨 Latest Troubling Messages")
distressed = df[df['is_distressed']]
st.dataframe(distressed[['datetime', 'student', 'question', 'intent']].sort_values('datetime', ascending=False))