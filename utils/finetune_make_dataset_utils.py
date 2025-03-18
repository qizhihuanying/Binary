import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.string_utils import *

np.random.seed(42)


def circular_rotate(series: list, n: int) -> list:
    n = n % len(series)  # Handle cases where n is larger than the length of the series
    return series[n:] + series[:n]


def get_positives(df: pd.DataFrame, pct_size: float) -> pd.DataFrame:
    tqdm.write("Generating positive pairs...")
    sample1 = list(tqdm(sample_from_text(df["original"].tolist(), pct_size), 
                        desc="Generating sample1 (positives)", unit="pair"))
    sample2 = list(tqdm(sample_from_text(df["original"].tolist(), pct_size), 
                        desc="Generating sample2 (positives)", unit="pair"))
    pos = [1] * len(sample1)
    return pd.DataFrame({"sample1": sample1, "sample2": sample2, "pos": pos})

def get_negatives(df: pd.DataFrame, pct_size: float) -> pd.DataFrame:
    sample1 = list(tqdm(sample_from_text(df["original"].tolist(), pct_size), 
                        desc="Generating sample1 (negatives)", unit="pair"))
    sample2 = list(tqdm(circular_rotate(sample_from_text(df["original"].tolist(), pct_size), 1), 
                        desc="Generating sample2 (negatives)", unit="pair"))
    pos = [0] * len(sample1)
    return pd.DataFrame({"sample1": sample1, "sample2": sample2, "pos": pos})


def get_truncated_df(df: pd.DataFrame, pct_size: float) -> pd.DataFrame:
    assert 0 < pct_size < 1  
    positives_df = get_positives(df, pct_size)
    negatives_df = get_negatives(df, pct_size)
    return pd.concat((positives_df, negatives_df))