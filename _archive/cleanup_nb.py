import json

path = 'notebooks/ML.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Define target snippets to identify cells for removal
redundant_snippets = [
    "## 1. Random Forest",
    "## 2. XGBoost",
    "## Đánh giá mô hình: Random Forest & XGBoost",
    "rf_pipeline = Pipeline(",  # Only the first ones
    "xgb_pipeline = Pipeline(", # Only the first ones
    "import polars as pl",      # Leftover broken code
    "name 'data' is not defined", # Broken markup/code from errors
    "X_train, X_test, y_train, y_test = train_test_split(\n    X, y,", # Early split
    "target = 'IS_PROFIT'\n\nfeatures = [", # Early feature selection
    "print(\"PurchaseNext90 value counts:\n\", rfm[\"PurchaseNext90\"].value_counts())" # Redundant print
]

# Keep track of indices to delete
# We need to be careful with "Pipeline" because it's used multiple times.
# We'll target cells by specific headers or unique logic.

indices_to_delete = []

for i, cell in enumerate(cells):
    source = "".join(cell.get('source', []))
    
    # 1. Remove early classification sections (Order-level)
    if "## 1. Random Forest" in source:
        # Delete this and the next few cells until we hit next major section
        j = i
        while j < len(cells) and "# RFM" not in "".join(cells[j].get('source', [])):
             # Wait, RFM is BEFORE this in some versions. 
             # Let's check the current order.
             pass

# RE-REFINING SCRIPT LOGIC based on line numbers from previous view_file
# Section 1: Order-Level Classification (Old)
# This starts at "## 1. Random Forest" (currently line 1431 in file view)
# and goes until the more comprehensive comparison starts.

# Let's use a more robust way: Find indices of specific section headers.

headers = {}
for i, cell in enumerate(cells):
    source = "".join(cell.get('source', []))
    if "# PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)" in source: headers['eda'] = i
    if "# RFM" in source: headers['rfm'] = i
    if "# Phân cụm" in source: headers['clustering'] = i
    if "## 1. Random Forest" in source: headers['old_rf'] = i
    if "### 1. Logistic Regression" in source: headers['new_models'] = i
    if "# TỔNG KẾT MÔ HÌNH (DASHBOARD)" in source: headers['dashboard'] = i

# Delete everything between 'old_rf' and 'new_models'
# This removes the redundant order-level RF and XGBoost blocks.
if 'old_rf' in headers and 'new_models' in headers:
    for idx in range(headers['old_rf'], headers['new_models']):
        indices_to_delete.append(idx)

# Consolidate RFM: 
# The predictive RFM block (headers['rfm']) is at Execution 21.
# The Clustering block (headers['clustering']) is at Execution 23.
# We should probably remove the redundant RFM calculation inside the clustering block.

for i, cell in enumerate(cells):
    source = "".join(cell.get('source', []))
    if "# 1. Tính toán các chỉ số RFM" in source and "df_rfm = df.copy()" in source:
        # This is the redundant one inside the clustering section. 
        # We will replace it with a shorter version that uses the existing 'rfm' df.
        cell['source'] = [
            "# Sử dụng dữ liệu RFM đã tính toán ở trên cho phân cụm\n",
            "from sklearn.cluster import KMeans\n",
            "from sklearn.preprocessing import StandardScaler\n\n",
            "# Chuẩn hóa dữ liệu\n",
            "features_model = ['Recency', 'Frequency', 'Monetary']\n",
            "scaler_cluster = StandardScaler()\n",
            "rfm_scaled = scaler_cluster.fit_transform(rf[features_model])\n\n",
            "# Phân cụm KMeans\n",
            "kmeans = KMeans(n_clusters=4, random_state=42)\n",
            "rf['Cluster'] = kmeans.fit_predict(rfm_scaled)\n\n",
            "# Thống kê cụm\n",
            "cluster_stats = rf.groupby('Cluster')[features_model + ['USERID']].agg({\n",
            "    'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'USERID': 'count'\n",
            "}).reset_index()\n",
            "print(cluster_stats)\n"
        ]

# Final deletion
new_cells = [c for i, c in enumerate(cells) if i not in indices_to_delete]
nb['cells'] = new_cells

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook cleanup complete.")
