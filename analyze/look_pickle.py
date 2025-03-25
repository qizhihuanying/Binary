import pickle

with open("datasets/miracl/zh/dev/processed_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data)  
