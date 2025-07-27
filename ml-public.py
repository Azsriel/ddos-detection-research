import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

### ---- PHASE 1: TRAIN ON PUBLIC DATASET ---- ###
# Load public dataset
# df = pd.read_csv("final_dataset.csv",nrows=100_000)  # Assume preprocessed flow file with labels
df = pd.read_parquet("CIC-DDoS2019-combined.parquet", engine="pyarrow")
df = df.dropna()


# Turn the Labels into Binary - Either Benign or DDoS
# Normalize label case
df["Label"] = df["Label"].str.strip().str.lower()

# Map all non-benign to "DDoS"
df["BinaryLabel"] = df["Label"].apply(lambda x: "Benign" if "benign" in x else "DDoS")


X = df.drop(columns=["Label", "BinaryLabel"])
y = df["BinaryLabel"]


'''
# If not converting to Binary then do this otherwise comment out
X = df.drop(columns=["Label"])
y = df["Label"]
'''

# Add unsupervised features
iso = IsolationForest(contamination=0.25)
df["anomaly_score"] = iso.fit_predict(X)

kmeans = KMeans(n_clusters=2)
df["cluster_id"] = kmeans.fit_predict(X)

# Augment data with new features
X_aug = pd.concat([X, df[["anomaly_score", "cluster_id"]]], axis=1)

# Train/Test Split
# Change to X instead of X_aug inside the function to get values for only C4.5
X_train, X_test, y_train, y_test = train_test_split(X_aug, y, test_size=0.99, random_state=42)

# Train C4.5-like model
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "trained_model.joblib")

# Offline evaluation
y_pred = clf.predict(X_test)
print("\n=== Public Dataset Evaluation ===")
print(classification_report(y_test, y_pred, digits=4))
