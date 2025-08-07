import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path

# ————— App Title —————
st.title("Participant Count by Age (Precision-Paths)")

# ————— Sidebar Controls —————
st.sidebar.header("Filter Settings")
min_age = st.sidebar.number_input("Min age", min_value=0, max_value=100, value=0)
max_age = st.sidebar.number_input("Max age", min_value=0, max_value=100, value=80)
incr    = st.sidebar.slider("Age bin size", min_value=1, max_value=20, value=5)
rest_only = st.sidebar.checkbox("Only include resting-state == True", value=True)

# ————— Load & Cache Data —————
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data_path = "data/data.csv"
df = load_data(data_path)

# ————— Parse Ages —————
def parse_age(age):
    if isinstance(age, str):
        s = age.strip().replace('–','-').replace('—','-').replace(' to ','-')
        nums = re.findall(r'\d+\.?\d*', s)
        if len(nums) >= 2:
            low, high = map(float, nums[:2])
            return (low + high) / 2
        if len(nums) == 1:
            return float(nums[0])
    try:
        return float(age)
    except:
        return np.nan

df['Age_num'] = df['Age'].apply(parse_age)

# ————— Apply Filters —————
df = df[df['Age_num'].notna()]
if rest_only:
    df = df[df['Mostly resting state/ Enough Resting State'] == True]
df = df[(df['Age_num'] >= min_age) & (df['Age_num'] < max_age + incr)]

# ————— Bin & Count —————
bins   = np.arange(min_age, max_age + incr, incr)
labels = [f"{b}–{b+incr}" for b in bins[:-1]]
df['age_bin'] = pd.cut(df['Age_num'], bins=bins, right=False,
                       labels=labels, include_lowest=True)

counts = df['age_bin'].value_counts().reindex(labels, fill_value=0)

# ————— Show Counts Table —————
st.subheader("Counts per Age Bin")
st.bar_chart(counts)  # built-in chart
st.dataframe(counts.rename_axis("Age Bin").reset_index(name="Count"))

# ————— Custom Matplotlib Plot —————
st.subheader("Bar Plot (Matplotlib)")
fig, ax = plt.subplots(figsize=(8, 4))
counts.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title("Participant Count by Age", fontsize=14)
ax.set_xlabel("Age Bin", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
