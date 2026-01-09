
import json
import os

notebook_path = '/Users/admin/ML/3A_Superstore/notebooks/ML.ipynb'

def add_rfm_section():
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 5. Phân cụm khách hàng (RFM Clustering)\n",
                "\n",
                "Phân tích **RFM** là một kỹ thuật tiếp thị được sử dụng để xác định các nhóm khách hàng tốt nhất bằng cách dựa trên các điểm số cho ba tiêu chí:\n",
                "- **Recency (R)**: Thời gian kể từ lần mua hàng cuối cùng.\n",
                "- **Frequency (F)**: Tần suất mua hàng.\n",
                "- **Monetary (M)**: Tổng số tiền đã chi tiêu."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.cluster import KMeans\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from datetime import datetime\n",
                "\n",
                "# 1. Chuẩn bị dữ liệu RFM\n",
                "df_rfm = df.copy()\n",
                "df_rfm['DATE_'] = pd.to_datetime(df_rfm['DATE_'], dayfirst=True)\n",
                "today = df_rfm['DATE_'].max()\n",
                "\n",
                "rfm_data = df_rfm.groupby('USERID').agg({\n",
                "    'DATE_': lambda x: (today - x.max()).days,\n",
                "    'ORDERID': 'count',\n",
                "    'TOTALPRICE': 'sum'\n",
                "}).reset_index()\n",
                "\n",
                "rfm_data.columns = ['USERID', 'Recency', 'Frequency', 'Monetary']\n",
                "print(\"Dữ liệu RFM (5 dòng đầu):\")\n",
                "print(rfm_data.head())\n",
                "\n",
                "# 2. Xử lý giá trị thiếu (nếu có)\n",
                "rfm_data = rfm_data.fillna(0)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Chuẩn hóa dữ liệu và Phân cụm\n",
                "scaler = StandardScaler()\n",
                "rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])\n",
                "\n",
                "# Sử dụng KMeans với k=4 (Ví dụ)\n",
                "kmeans = KMeans(n_clusters=4, random_state=42)\n",
                "rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)\n",
                "\n",
                "# 4. Thống kê theo cụm\n",
                "cluster_stats = rfm_data.groupby('Cluster').agg({\n",
                "    'Recency': 'mean',\n",
                "    'Frequency': 'mean',\n",
                "    'Monetary': 'mean',\n",
                "    'USERID': 'count'\n",
                "}).reset_index()\n",
                "\n",
                "cluster_stats.columns = ['Cluster', 'AvgRecency', 'AvgFrequency', 'AvgMonetary', 'CustomerCount']\n",
                "print(\"Thống kê theo từng cụm khách hàng:\")\n",
                "print(cluster_stats)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Trực quan hóa kết quả phân cụm\n",
                "import seaborn as sns\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "plt.figure(figsize=(12, 6))\n",
                "plt.subplot(1, 2, 1)\n",
                "sns.scatterplot(data=rfm_data, x='Recency', y='Monetary', hue='Cluster', palette='viridis')\n",
                "plt.title('Recency vs Monetary')\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "sns.scatterplot(data=rfm_data, x='Frequency', y='Monetary', hue='Cluster', palette='viridis')\n",
                "plt.title('Frequency vs Monetary')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Giải thích các cụm khách hàng (Cluster Interpretation)\n",
                "\n",
                "Dựa trên bảng thống kê ở trên, chúng ta có thể phân loại khách hàng:\n",
                "- **Khách hàng VIP**: Recency thấp, Frequency cao, Monetary cao.\n",
                "- **Khách hàng tiềm năng**: Recency thấp, Frequency trung bình.\n",
                "- **Khách hàng mới**: Recency thấp nhưng Frequency và Monetary còn thấp.\n",
                "- **Khách hàng rời bỏ**: Recency rất cao, lâu rồi không quay lại mua hàng."
            ]
        }
    ]

    nb['cells'].extend(new_cells)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    add_rfm_section()
    print("RFM section added to notebook successfully.")
