import json
import os

notebook_path = '/Users/admin/ML/3A_Superstore/notebooks/ML.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Globals and ID/NumItems fix (already mostly done, but let's be sure)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            new_line = line.replace('CATEGORY1', 'ITEMID').replace('NumCategories', 'NumItems')
            new_source.append(new_line)
        cell['source'] = new_source

# 2. Extract the label creation cell and RFM definition cell
rfm_def_idx = -1
label_cell_idx = -1
clustering_idx = -1

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if 'df_rfm_clean = df.dropna' in source_str:
            rfm_def_idx = i
        if 'rfm = rfm.merge(future_summary' in source_str:
            label_cell_idx = i
        if 'kmeans = KMeans' in source_str:
            clustering_idx = i

# 3. Move label creation cell BEFORE clustering if it's after
if label_cell_idx != -1 and clustering_idx != -1 and label_cell_idx > clustering_idx:
    label_cell = nb['cells'].pop(label_cell_idx)
    # Re-find clustering index after pop
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'kmeans = KMeans' in "".join(cell['source']):
            clustering_idx = i
            break
    nb['cells'].insert(clustering_idx, label_cell)
    print(f"Moved label creation cell from {label_cell_idx} to {clustering_idx}")

# 4. Fix variable scope in RFM definition cell (ensure rfm is initialized even if empty)
if rfm_def_idx != -1:
    source = nb['cells'][rfm_def_idx]['source']
    # Ensure rfm is initialized outside the if block or available
    new_source = [
        "    # Filter out rows with missing DATE_ for RFM analysis\n",
        "    df_rfm_clean = df.dropna(subset=[\"DATE_\"]).copy()\n",
        "    \n",
        "    # Initialize rfm as empty in case the block doesn't run\n",
        "    rfm = pd.DataFrame()\n",
        "    \n",
        "    if not df_rfm_clean.empty:\n",
        "        reference_date = df_rfm_clean[\"DATE_\"].max() - pd.Timedelta(days=365)\n",
        "    \n",
        "        # Dữ liệu quá khứ\n",
        "        past_df = df_rfm_clean[df_rfm_clean[\"DATE_\"] <= reference_date].copy()\n",
        "    \n",
        "        # Tạo RFM từ quá khứ\n",
        "        rfm = past_df.groupby(\"USERID\").agg(\n",
        "            Recency=(\"DATE_\", lambda x: (reference_date - x.max()).days),\n",
        "            Frequency=(\"ORDERID\", \"nunique\"),\n",
        "            Monetary=(\"TOTALPRICE\", \"sum\"),\n",
        "            AvgBasketSize=(\"TOTALPRICE\", \"mean\"),\n",
        "            NumItems=(\"ITEMID\", \"nunique\")\n",
        "        ).reset_index()\n",
        "    else:\n",
        "        print(\"No valid dates found for RFM calculation.\")\n"
    ]
    nb['cells'][rfm_def_idx]['source'] = new_source
    nb['cells'][rfm_def_idx]['outputs'] = []
    nb['cells'][rfm_def_idx]['execution_count'] = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
