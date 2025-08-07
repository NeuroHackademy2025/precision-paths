#8/7/2025 Elizabeth Li & Jonathan Ahern @ Neurohackademy
# app to deploy on steamlit

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
from html import escape

# ————————————————————— App Title —————————————————————
title = "PFM Data Explorer"
st.set_page_config(page_title=title)  # browser tab title (optional)

st.markdown(
    f"<h1 style='text-align:center;'>{escape(title)}</h1>",
    unsafe_allow_html=True
)

# ————————————————————— LOAD AND CACHE DATA —————————————————————
# @st.cache_data  # Temporarily disable caching while you iterate
def load_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip() #remove white space

    # map empty/NaN/whitespace cells to Unknown
    def empty_cell_to_unknown(s):
        return(
            s.astype("string").str.strip()
            .replace(r"^\s*$", pd.NA, regex=True) # blank -> NA
            .fillna("Unknown") # NA -> "Unknown"
        )
    # select which columns to add "Unknowns" to
    for col in ["Sex", "Handedness"]:
        if col in df.columns:
            df[col] = empty_cell_to_unknown(df[col])
    
    # parse Age_num
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

    # parse Study Year to a 4-digit int
    def parse_year(y):
        m = re.search(r'\b(\d{4})\b', str(y))
        return int(m.group(1)) if m else np.nan
    df['StudyYear_num'] = df['Study Year'].apply(parse_year)
    return df

df = load_data("data/data.csv")

# ————————————————————— SIDEBAR —————————————————————
st.sidebar.header("Select demographics")

# 1. age range and bin size
age_min  = int(df['Age_num'].min(skipna=True))
age_max  = int(df['Age_num'].max(skipna=True))
age_range = age_max - age_min

bin_size = st.sidebar.number_input(
    "Age bin size",
    min_value=1,
    max_value=age_range,      
    step=1,
    value=5,
    help="Width of each age bin in years"
)
age_range = st.sidebar.slider(
    "Age range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
    step=1,
    help="Minimum and maximum are constrained by the minimum and maximum ages available"
)

# 2 and 3. sex and handedness checkboxes ("pills")
def multi_pill_filter(label, options, help_text=None):
    """Render a multi-select pill widget and return the list of selected options"""
    return st.sidebar.pills(
        label,
        options=options,
        selection_mode='multi',
        default=[],
        help=help_text or f"Filter by {label.lower()}"
    )

sex_options=["Male", "Female", "Other", "Unknown"]
handedness_options=["Left", "Right", "Ambidextrous", "Unknown"]

sex_filters = multi_pill_filter("Sex", sex_options)
handedness_filters = multi_pill_filter("Handedness", handedness_options)

# 4. study year
year_min = int(df['StudyYear_num'].min(skipna=True))
year_max = int(df['StudyYear_num'].max(skipna=True))

study_year_range = st.sidebar.slider(
    "Study year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1,
    help="Minimum and maximum are constrained by the minimum and maximum study years available"
)

st.sidebar.header("Select dataset")

# ————————————————————— APPLY FILTERS —————————————————————
filtered = df.copy()

#1. age 
filtered = filtered[
    (filtered['Age_num'] >= age_range[0]) &
    (filtered['Age_num'] <= age_range[1])
]

#2. sex
if sex_filters:
    filtered = filtered[filtered['Sex'].isin(sex_filters)]

#3. handedness
if handedness_filters:
    filtered = filtered[filtered["Handedness"].isin(handedness_filters)]

#4. study year
filtered = filtered[
    (filtered['StudyYear_num'] >= study_year_range[0]) &
    (filtered['StudyYear_num'] <= study_year_range[1])
]

# ————————————————————— BINNING —————————————————————
bins = np.arange(age_range[0], age_range[1]+bin_size, bin_size)
# say bin size = 5, produce labels on the x-axis like "0-5", "5-10", ..., "75-80"
labels = [f"{b}–{b+bin_size}" for b in bins[:-1]]
filtered['age_bin'] = pd.cut(
    filtered['Age_num'],
    bins=bins,
    right=False,
    labels=labels,
    include_lowest=True
)
counts = filtered['age_bin'].value_counts().reindex(labels, fill_value=0)

# ————————————————————— DISPLAY HISTOGRAM —————————————————————
st.subheader("Counts per Age Bin")
st.bar_chart(counts)

# ————— DISPLAY DEMOGRAPHICS —————
st.subheader("Group Demographics")
st.write(f"Total participants: {len(filtered)}")
st.write(f"Age range: {filtered['Age_num'].min()}–{filtered['Age_num'].max()} years (Mean: {filtered['Age_num'].mean():.3f} ± {filtered['Age_num'].std():.3f} years)")
st.write(f'Sex: {(df['Sex'] == 'Female').sum()} Female')

# ————— DISPLAY DATA CSV —————
st.subheader("Data")

# Make a display-only copy
display_df = filtered.reset_index(drop=True).copy()

# turn empty cells into "Unknown" for display only
obj_cols = display_df.select_dtypes(include=["object", "string", "category"]).columns
if len(obj_cols) > 0:
    display_df[obj_cols] = (
        display_df[obj_cols]
            .apply(lambda s: s.astype("string").str.strip())
            .replace(r"^\s*$", pd.NA, regex=True)
    )

na_mask = display_df.isna()
display_df = display_df.astype(str)
display_df = display_df.where(~na_mask, "Unknown")

# render
st.dataframe(display_df, use_container_width=True)