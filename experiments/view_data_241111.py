import pickle
import numpy as np

file_path = "./data/cl_n20_241118_162131.pkl"
# file_path = "./data/cl_n8_241117_180006.pkl"

with open(file_path, "rb") as f:
    results = pickle.load(f)
    seeds = pickle.load(f)
    env_config = pickle.load(f)
    time_taken = pickle.load(f)

print("Pause here and see if...")
