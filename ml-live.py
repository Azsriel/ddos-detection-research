import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

### ---- PHASE 1: TRAIN ON PUBLIC DATASET ---- ###
# Load public dataset
# df = pd.read_csv("cicddos2019.csv")  # Assume preprocessed flow file with labels
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
iso = IsolationForest(contamination=0.1)
df["anomaly_score"] = iso.fit_predict(X)

kmeans = KMeans(n_clusters=2)
df["cluster_id"] = kmeans.fit_predict(X)

# Augment data with new features
X_aug = pd.concat([X, df[["anomaly_score", "cluster_id"]]], axis=1)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_aug, y, test_size=0.3, random_state=42)

# Train C4.5-like model
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "trained_model.joblib")

# Offline evaluation
y_pred = clf.predict(X_test)
print("\n=== Public Dataset Evaluation ===")
print(classification_report(y_test, y_pred))


# Save list of training feature names
with open("features.txt", "w") as f:
    for col in X.columns:
        f.write(f"{col}\n")

# Load trained model
clf = joblib.load("trained_model.joblib")

# Add unsupervised features
iso = IsolationForest(contamination=0.1)
df["anomaly_score"] = iso.fit_predict(X)

kmeans = KMeans(n_clusters=2)
df["cluster_id"] = kmeans.fit_predict(X)


# Load testbed traffic processed with CICFlowMeter
live_df = pd.read_csv("mixed_traffic.pcap_Flow.csv")  # Output from CICFlowMeter
live_df = live_df.dropna()

X_live = live_df.drop(columns=["Label"])  # Label may not be present, or optional

# Load training features list
with open("features.txt", "r") as f:
    expected_features = [line.strip() for line in f.readlines()]

# Drop extra cols, fill missing with 0
for col in expected_features:
    if col not in X_live.columns:
        X_live[col] = 0

X_live = X_live[expected_features]

# Add same unsupervised features
live_df["anomaly_score"] = iso.predict(X_live)
live_df["cluster_id"] = kmeans.predict(X_live)

X_live_aug = pd.concat([X_live, live_df[["anomaly_score", "cluster_id"]]], axis=1)
'''
# If testbed traffic has labels, compute metrics
if "Label" in live_df.columns:
    y_live = live_df["Label"]
    y_live_pred = clf.predict(X_live_aug)
    print("\n=== Live Testbed Evaluation ===")
    print(classification_report(y_live, y_live_pred))
else:
    # If no label, just predict and print classes
    y_live_pred = clf.predict(X_live_aug)
    print("\n=== Live Predictions ===")
    print(pd.Series(y_live_pred).value_counts())
'''

# Assume no labels whenever
y_live_pred = clf.predict(X_live_aug)
print("\n=== Live Predictions ===")
print(pd.Series(y_live_pred).value_counts())

