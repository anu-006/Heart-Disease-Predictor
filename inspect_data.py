import pickle

# inspect the dataset file (now named data.pkl)
with open("data.pkl","rb") as f:
    data = pickle.load(f)

print("Columns:", data.columns.tolist())
print("Shape:", data.shape)
print("Head:")
print(data.head())