# Import libraries
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load log CSV file
df = pd.read_csv('./log/log.csv', parse_dates=['datetime'])
st.title("📊 Chatbot Statistics Dashboard")

# Intent type distribution map
st.header("✨ FAQ Intent Type Distribution")
intent_counts = df['intent'].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(intent_counts.index, intent_counts.values, color='skyblue')
ax.set_xlabel('Intent Type')
ax.set_ylabel('Count')
ax.set_xticklabels(intent_counts.index, rotation=0) # Display x-axis labels horizontally
ax.set_title('Intent Type Distribution')
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
st.header("🚨 Latest Troubling Message")
distressed = df[df['is_distressed']]
st.dataframe(distressed[['datetime', 'student', 'question', 'intent']].sort_values('datetime', ascending=False))