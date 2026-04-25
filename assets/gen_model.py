import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import to_onnx

# Ensure output directory exists
os.makedirs("generated", exist_ok=True)

# 1. Train a simple model
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Small model: 3 trees, max depth 3 (easy to inspect)
clf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 2. Convert to ONNX
# We must specify the input type (float32 tensor)
onx = to_onnx(clf, X[:1])

# 3. Save to generated/ directory
output_path = "generated/model.onnx"
with open(output_path, "wb") as f:
    f.write(onx.SerializeToString())

print(f"✅ Generated '{output_path}' with {len(clf.estimators_)} trees.")
