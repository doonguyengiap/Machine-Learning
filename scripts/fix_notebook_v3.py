import json

notebook_path = '/Users/admin/ML/3A_Superstore/notebooks/ML.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Consolidated RFM and Labeling logic
consolidated_rfm_source = [
    "# Filter out rows with missing DATE_ for RFM analysis\n",
    "df_rfm_clean = df.dropna(subset=[\"DATE_\"]).copy()\n",
    "\n",
    "if not df_rfm_clean.empty:\n",
    "    # 1. Set reference date\n",
    "    reference_date = df_rfm_clean[\"DATE_\"].max() - pd.Timedelta(days=365)\n",
    "    \n",
    "    # 2. Historical Data (for Features)\n",
    "    past_df = df_rfm_clean[df_rfm_clean[\"DATE_\"] <= reference_date].copy()\n",
    "    \n",
    "    # Create RFM features\n",
    "    rfm = past_df.groupby(\"USERID\").agg(\n",
    "        Recency=(\"DATE_\", lambda x: (reference_date - x.max()).days),\n",
    "        Frequency=(\"ORDERID\", \"nunique\"),\n",
    "        Monetary=(\"TOTALPRICE\", \"sum\"),\n",
    "        AvgBasketSize=(\"TOTALPRICE\", \"mean\"),\n",
    "        NumItems=(\"ITEMID\", \"nunique\")\n",
    "    ).reset_index()\n",
    "    \n",
    "    # 3. Future Data (for Labels - Target 90 days)\n",
    "    future_df = df_rfm_clean[(df_rfm_clean[\"DATE_\"] > reference_date) & \n",
    "                   (df_rfm_clean[\"DATE_\"] <= reference_date + pd.Timedelta(days=90))].copy()\n",
    "    \n",
    "    future_summary = future_df.groupby(\"USERID\", as_index=False).agg(\n",
    "        FutureOrders90=(\"ORDERID\", \"nunique\"),\n",
    "        FutureTotal90=(\"TOTALPRICE\", \"sum\")\n",
    "    )\n",
    "    \n",
    "    # 4. Merge and Handle Labels\n",
    "    rfm = rfm.merge(future_summary, on=\"USERID\", how=\"left\").fillna(0)\n",
    "    rfm[\"PurchaseNext90\"] = (rfm[\"FutureOrders90\"] > 0).astype(int)\n",
    "    rfm[\"Churn90\"] = (rfm[\"FutureOrders90\"] == 0).astype(int)\n",
    "    \n",
    "    # CLV 90 day labels\n",
    "    import numpy as np\n",
    "    rfm[\"Frequency_adj\"] = rfm[\"Frequency\"].replace(0,1)\n",
    "    rfm[\"CLV90_raw\"] = rfm[\"FutureTotal90\"]\n",
    "    rfm[\"CLV90_capped\"] = np.clip(rfm[\"CLV90_raw\"], 0, rfm[\"CLV90_raw\"].quantile(0.99))\n",
    "    rfm[\"CLV90_log\"] = np.log1p(rfm[\"CLV90_capped\"])\n",
    "    \n",
    "    print(f\"RFM table created with {len(rfm)} users.\")\n",
    "    print(\"PurchaseNext90 value counts:\\n\", rfm[\"PurchaseNext90\"].value_counts())\n",
    "else:\n",
    "    print(\"No valid dates found for RFM calculation.\")\n",
    "    rfm = pd.DataFrame()\n"
]

# Indices to remove
to_remove = []
inserted = False

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Identify the original RFM and label cells
        if ('df_rfm_clean = df.dropna' in source_str or 
            'rfm = rfm.merge(future_summary' in source_str or 
            'reference_date = df["DATE_"].max()' in source_str):
            
            if not inserted:
                cell['source'] = consolidated_rfm_source
                cell['outputs'] = []
                cell['execution_count'] = None
                cell['metadata'] = {"id": "consolidated_rfm_logic"}
                inserted = True
                print(f"Consolidated logic into cell {i}")
            else:
                to_remove.append(i)

# Clustering cell - it uses 'rfm_data', we should make sure it works with our 'rfm'
# Let's find the clustering cell and ensure it doesn't collide or uses the correct data
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if 'kmeans = KMeans' in source_str:
            # Optionally update this cell to use 'rfm' if needed, 
            # but usually it's better to keep it separate if it works.
            pass

# Remove redundant cells (reverse to keep indices valid)
for i in sorted(to_remove, reverse=True):
    nb['cells'].pop(i)
    print(f"Removed redundant cell at index {i}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
