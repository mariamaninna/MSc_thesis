import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from clemcore.clemeval import save_clem_table


root = Path(r"C:\Users\white\Desktop\MSc_thesis\self_refine\refine_only")
STATIC = ["bbh", "cladder", "ifeval", "mmlu_pro", "eqbench"]

def subsample_static(root_dir):
    """
    The static benchmark scores need to be analyzed separately from the game scores.
    """
    raw_table = pd.read_csv(Path(root_dir) / "raw.csv", index_col=0)
    is_static = raw_table["game"].isin(STATIC)
    static_rows = raw_table[is_static == True]
    game_rows = raw_table[is_static == False]
    save_clem_table(static_rows, r"C:\Users\white\Desktop\MSc_thesis\self_refine\refine_only\static")
    save_clem_table(game_rows, r"C:\Users\white\Desktop\MSc_thesis\self_refine\refine_only\games")

subsample_static(root)