
import json
import os

notebook_path = '/Users/admin/ML/3A_Superstore/notebooks/ML.ipynb'

def add_forecast_section():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 4. Dự báo Doanh thu 6 tháng (6-month Revenue Forecasting)\n",
                "\n",
                "Trong phần này, chúng ta sẽ sử dụng **Linear Regression** và **KNN Regressor** để dự báo doanh thu trong 6 tháng tới dựa trên **Vùng kinh doanh (REGION)** và **Sản phẩm (ITEMCODE)**.\n",
                "\n",
                "**Lưu ý:** Khác với Logistic Regression (phân loại khách hàng/lợi nhuận), ở đây chúng ta dự đoán giá trị doanh thu cụ thể (biến liên tục), do đó cần sử dụng các thuật toán Regression."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.neighbors import KNeighborsRegressor\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                "from sklearn.model_selection import train_test_split\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "# 1. Chuẩn bị dữ liệu Regression\n",
                "# Chúng ta cần gộp dữ liệu theo Tháng, Vùng và Sản phẩm\n",
                "df_reg = df.copy()\n",
                "df_reg['DATE_'] = pd.to_datetime(df_reg['DATE_'], dayfirst=True)\n",
                "df_reg['Year'] = df_reg['DATE_'].dt.year\n",
                "df_reg['Month'] = df_reg['DATE_'].dt.month\n",
                "\n",
                "# Tính TimeIndex (số tháng tính từ mốc bắt đầu)\n",
                "min_year = df_reg['Year'].min()\n",
                "df_reg['TimeIndex'] = (df_reg['Year'] - min_year) * 12 + df_reg['Month']\n",
                "\n",
                "# Group by để lấy doanh thu hàng tháng theo Region và Item\n",
                "monthly_data = df_reg.groupby(['REGION', 'ITEMCODE', 'Year', 'Month', 'TimeIndex'])['TOTALPRICE'].sum().reset_index()\n",
                "\n",
                "print(f\"Kích thước dữ liệu sau khi gộp: {monthly_data.shape}\")\n",
                "monthly_data.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Huấn luyện và So sánh Model\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
                "from sklearn.pipeline import Pipeline\n",
                "\n",
                "X_reg = monthly_data[['REGION', 'ITEMCODE', 'TimeIndex']]\n",
                "y_reg = monthly_data['TOTALPRICE']\n",
                "\n",
                "X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)\n",
                "\n",
                "preprocessor_reg = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('cat', OneHotEncoder(handle_unknown='ignore'), ['REGION', 'ITEMCODE']),\n",
                "        ('num', StandardScaler(), ['TimeIndex'])\n",
                "    ]\n",
                ")\n",
                "\n",
                "# Pipeline cho Linear Regression\n",
                "lr_reg_pipeline = Pipeline([\n",
                "    ('preprocess', preprocessor_reg),\n",
                "    ('model', LinearRegression())\n",
                "])\n",
                "\n",
                "# Pipeline cho KNN Regressor\n",
                "knn_reg_pipeline = Pipeline([\n",
                "    ('preprocess', preprocessor_reg),\n",
                "    ('model', KNeighborsRegressor(n_neighbors=5))\n",
                "])\n",
                "\n",
                "def evaluate_regression(model, X_train, y_train, X_test, y_test, name):\n",
                "    model.fit(X_train, y_train)\n",
                "    y_pred = model.predict(X_test)\n",
                "    print(f\"--- Kết quả {name} ---\")\n",
                "    print(f\"R2 Score: {r2_score(y_test, y_pred):.4f}\")\n",
                "    print(f\"MAE: {mean_absolute_error(y_test, y_pred):.2f}\")\n",
                "    print(f\"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}\\n\")\n",
                "    return model\n",
                "\n",
                "lr_reg_model = evaluate_regression(lr_reg_pipeline, X_train_r, y_train_r, X_test_r, y_test_r, \"Linear Regression\")\n",
                "knn_reg_model = evaluate_regression(knn_reg_pipeline, X_train_r, y_train_r, X_test_r, y_test_r, \"KNN Regressor\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Dự báo 6 tháng tới (6-month Forecast)\n",
                "def forecast_next_6_months(region, item_code, model):\n",
                "    latest_time = monthly_data['TimeIndex'].max()\n",
                "    future_times = pd.DataFrame({\n",
                "        'REGION': [region] * 6,\n",
                "        'ITEMCODE': [item_code] * 6,\n",
                "        'TimeIndex': [latest_time + i for i in range(1, 7)]\n",
                "    })\n",
                "    \n",
                "    predictions = model.predict(future_times)\n",
                "    total_forecast = predictions.sum()\n",
                "    \n",
                "    print(f\"Dự báo doanh thu cho Region: {region}, Item: {item_code} trong 6 tháng tới:\")\n",
                "    for i, p in enumerate(predictions):\n",
                "        print(f\" Tháng {i+1}: {max(0, p):.2f}\")\n",
                "    print(f\" Tổng Doanh thu dự kiến: {max(0, total_forecast):.2f}\")\n",
                "\n",
                "# Demo dự báo với Linear Regression\n",
                "if not monthly_data.empty:\n",
                "    sample_row = monthly_data.iloc[0]\n",
                "    forecast_next_6_months(sample_row['REGION'], sample_row['ITEMCODE'], lr_reg_model)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### So sánh chi tiết các phương pháp (Detailed Comparison)\n",
                "\n",
                "| Đặc điểm | Logistic Regression | Linear Regression | KNN Regressor |\n",
                "| :--- | :--- | :--- | :--- |\n",
                "| **Mục tiêu (Output)** | Phân loại (0/1, High Value, Profitability) | Giá trị số liên tục (Revenue, Price) | Giá trị số liên tục |\n",
                "| **Loại bài toán** | Classification | Regression | Regression |\n",
                "| **Ưu điểm** | Dễ giải thích xác suất, hiệu quả với dữ liệu tách biệt tuyến tính. | Rất nhanh, là baseline mạnh cho xu hướng thời gian (trend). | Linh hoạt, nắm bắt được các quan hệ phi tuyến tính cục bộ. |\n",
                "| **Nhược điểm** | Không thể dự đoán giá trị cụ thể của doanh thu. | Khó nắm bắt các dao động phức tạp nếu quan hệ không tuyến tính. | Nhạy cảm với nhiễu và cần nhiều dữ liệu để chính xác. |\n",
                "\n",
                "**Kết luận:**\n",
                "Để dự báo doanh thu 6 tháng cho từng sản phẩm và vùng, **Linear Regression** cung cấp cái nhìn về xu hướng tăng/giảm theo thời gian, trong khi **KNN Regressor** hữu ích nếu doanh số phụ thuộc nhiều vào các mẫu (patterns) tương tự trong quá khứ của các sản phẩm cùng loại."
            ]
        }
    ]

    nb['cells'].extend(new_cells)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    add_forecast_section()
    print("Forecast section added to notebook.")
