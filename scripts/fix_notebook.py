import json
import os

notebook_path = '/Users/admin/ML/3A_Superstore/notebooks/ML.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Check if conversion cell exists, if not, insert it
has_conv = any('fix_date_conversion' in str(cell.get('metadata', {}).get('id', '')) for cell in nb['cells'])
if not has_conv:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": "fix_date_conversion"
        },
        "outputs": [],
        "source": [
            "# Convert DATE_ to datetime early on to avoid type errors\n",
            "import pandas as pd\n",
            "df['DATE_'] = pd.to_datetime(df['DATE_'], errors='coerce')\n",
            "print(f\"Total records: {len(df)}\")\n",
            "print(f\"Number of missing dates: {df['DATE_'].isna().sum()}\")"
        ]
    }
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and any('robust_read_processed' in line for line in cell['source']):
            nb['cells'].insert(i + 1, new_cell)
            break

# 2. Globally replace CATEGORY1 with ITEMID in all code cells
# and NumCategories with NumItems
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            new_line = line.replace('CATEGORY1', 'ITEMID').replace('NumCategories', 'NumItems')
            new_source.append(new_line)
        cell['source'] = new_source

# 3. Ensure RFM logic has null checks (if not already there)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if 'reference_date =' in source_str and 'df_rfm_clean' not in source_str:
             # This is an old version of the cell, replace it
             cell['source'] = [
                "    # Filter out rows with missing DATE_ for RFM analysis\n",
                "    df_rfm_clean = df.dropna(subset=[\"DATE_\"]).copy()\n",
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
                "        print(\"No valid dates found for RFM calculation.\")\n",
                "        rfm = pd.DataFrame(columns=[\"USERID\", \"Recency\", \"Frequency\", \"Monetary\", \"AvgBasketSize\", \"NumItems\"])\n"
            ]
             cell['outputs'] = []
             cell['execution_count'] = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
