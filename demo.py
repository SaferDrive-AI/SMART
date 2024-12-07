import pickle
import torch
from pathlib import Path
import glob

# Define base directories
base_dir_1 = (
    "/home/haoweis/trajdata_smart/SMART/data/waymo_processed/scenario/validation"
)
base_dir_2 = "/home/haoweis/trajdata_smart/SMART/data/smart_format"

# Iterate through all pkl files in both directories
for pkl_path in glob.glob(f"{base_dir_1}/*.pkl"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"Processing {pkl_path}")
    print(data.keys())

for pkl_path in glob.glob(f"{base_dir_2}/*.pkl"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"Processing {pkl_path}")
    print(data.keys())

print("aaa")
