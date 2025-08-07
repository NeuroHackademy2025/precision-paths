#8/6/2025 Elizabeth Li @ Neurohackademy
# script to make figure for precision-paths project
# x-axis: age of individual participants
# y-axis: count

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

data_path = 'data/data.csv'
rest_col = 'Mostly resting state/ Enough Resting State'
min_age, max_age, incr = 0, 80, 5

df = pd.read_csv(data_path)

def parse_age(age):
    # Handle strings first
    if isinstance(age, str):
        # Normalize various range separators to a simple hyphen
        s = age.strip().replace('–', '-').replace('—', '-').replace(' to ', '-')
        # Look for exactly two numbers (a range)
        nums = re.findall(r'\d+\.?\d*', s)
        if len(nums) >= 2:
            low, high = map(float, nums[:2])
            return (low + high) / 2
        # If only one number is found, return it
        if len(nums) == 1:
            return float(nums[0])
    # Fallback: try direct float conversion
    try:
        return float(age)
    except Exception:
        return np.nan

df['Age_num'] = df['Age'].apply(parse_age)
 
# print("Sample conversions (Age → Age_num):")
# print(df[['Age', 'Age_num']].drop_duplicates().head(10), "\n")
# print("Total rows, failures to parse:", len(df), df['Age_num'].isna().sum(), "\n")


# ─── FILTER ─────────────────────────────────────────────────────────
# 1) Must parse age
df = df[df['Age_num'].notna()]

# 2) Must meet resting-state criterion
df = df[df[rest_col] == True]

# 3) (Optional) keep only within your desired span:
df = df[(df['Age_num'] >= min_age) & (df['Age_num'] < max_age + incr)]


# ─── BINNING ────────────────────────────────────────────────────────
# edges: [10,15,20,...,80]
bins = np.arange(min_age, max_age + incr, incr)
labels = [f"{b}–{b+incr}" for b in bins[:-1]]

df['age_bin'] = pd.cut(
    df['Age_num'],
    bins=bins,
    right=False,
    labels=labels,
    include_lowest=True
)

# print counts
counts = df['age_bin'].value_counts().reindex(labels, fill_value=0)
# print("Counts per bin:")
# print(counts, "\n")

# plot
fig, ax = plt.subplots(figsize=(8,6))
counts.plot(kind='bar', ax=ax, color='steelblue')

ax.set_title("Participant Count by Age", fontsize=16)
ax.set_xlabel("Age (years)", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Get a sorted list of all non‐NaN ages
# ages_sorted = sorted(df['Age_num'].dropna())

# # Print them
# print("All ages (parsed), in increasing order:")
# print(ages_sorted)
# print(len(ages_sorted))