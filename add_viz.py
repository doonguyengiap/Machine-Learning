
import json

notebook_path = '/Users/admin/ML/3A_Superstore/notebooks/ML.ipynb'

def add_viz_cell():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    viz_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Plotting the comparison\n",
            "plt.figure(figsize=(12, 6))\n",
            "df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')\n",
            "sns.barplot(data=df_melted, x='Score', y='Model', hue='Metric')\n",
            "plt.title('Model Comparison - Accuracy, F1-score, ROC-AUC')\n",
            "plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
            "plt.show()"
        ]
    }

    # Find the cell with results_df and insert the viz cell after it
    insert_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'results_df = pd.DataFrame' in source:
                insert_idx = i + 1
                break
    
    if insert_idx != -1:
        nb['cells'].insert(insert_idx, viz_cell)
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Visualization cell added.")
    else:
        print("Could not find results_df cell.")

if __name__ == '__main__':
    add_viz_cell()
