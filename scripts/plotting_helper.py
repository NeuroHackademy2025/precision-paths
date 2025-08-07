import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_age(age):
    """
    Convert various age formats into a numeric value.
    """

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