# 3A Superstore Machine Learning Project

Dự án phân tích dữ liệu và học máy toàn diện dựa trên bộ dữ liệu Superstore, bao gồm quy trình xử lý dữ liệu tự động, huấn luyện và so sánh nhiều mô hình, và ứng dụng Web Dashboard để trực quan hóa kết quả.

## 1. Cấu trúc thư mục (Repository Structure)

```
3A_Superstore/
├── README.md                # Hướng dẫn chi tiết
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── data/                    # Chứa dữ liệu
│   └── processed_data.csv   # Dữ liệu sau khi tiền xử lý
├── src/                     # Mã nguồn Pipeline (Data & ML)
│   ├── preprocessing.py     # Làm sạch và tiền xử lý dữ liệu
│   ├── eda.py               # Exploratory Data Analysis
│   ├── feature_engineering.py # Tạo đặc trưng (RFM, Seasonality)
│   ├── models.py            # Định nghĩa các mô hình (Logistic, RF, XGBoost...)
│   ├── evaluation.py        # Đánh giá và so sánh mô hình
│   ├── forecasting.py       # Dự báo doanh thu chuỗi thời gian
│   └── main.py              # Script chạy toàn bộ pipeline ML
├── webapp/                  # Mã nguồn Web Application (Flask)
│   ├── app.py               # Backend Flask Server
│   └── templates/           # Giao diện HTML (Dashboard, Predictions)
├── notebooks/               # Jupyter Notebooks cho phân tích thử nghiệm
│   ├── ML.ipynb             # Notebook chính (EDA)
│   └── model_experiments.ipynb # Training & Tuning Models (Manual Playground)
└── reports/                 # Kết quả output
    ├── figures/             # Biểu đồ phân tích
    └── model_metrics.json   # Kết quả đánh giá mô hình
```

## 2. Hướng dẫn cài đặt (Installation)

Yêu cầu Python 3.8+. Khuyến nghị sử dụng môi trường ảo.

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate   # MacOS/Linux
# .venv\Scripts\activate    # Windows

# 2. Cài đặt thư viện
pip install -r requirements.txt
```

## 3. Web Application & Dashboard

Dự án đi kèm với một Web Dashboard hiện đại để xem báo cáo và truy vấn kết quả.

**Cách chạy:**
```bash
python webapp/app.py
```

Sau khi server khởi động, truy cập trình duyệt tại: `http://localhost:5001`

**Tính năng nổi bật:**
- **Store Data**: Xem dữ liệu thô và đã xử lý.
- **Analytics Dashboard**:
    - **Age Pyramid by Gender**: Phân tích nhân khẩu học chi tiết (Tháp tuổi).
    - **Pareto Analysis (80/20)**: Nhận diện nhóm khách hàng "Vàng" mang lại 80% doanh thu.
    - **Revenue Forecasting**: Dự báo doanh thu 3-6 tháng tới.
- **Prediction**: Tải lên file CSV mới để dự đoán khách hàng tiềm năng.

## 4. Chạy Pipeline Học Máy (Machine Learning Pipeline)

Để chạy lại toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện mô hình:

```bash
python src/main.py
```

**Các bước xử lý chính:**
1.  **Preprocessing**: Tự động xử lý missing values, chuẩn hóa định dạng ngày tháng và tiền tệ.
2.  **Feature Engineering**:
    - Tính toán RFM (Recency, Frequency, Monetary).
    - Tạo biến mục tiêu `IS_PROFIT` (Lợi nhuận/Không lợi nhuận).
3.  **Modeling**:
    - So sánh 10+ mô hình: Logistic Regression, Random Forest, XGBoost, KNN, Naive Bayes...
    - Tự động cân bằng dữ liệu (Class Weighting) để xử lý mất cân bằng mẫu.
4.  **Evaluation**: Xuất báo cáo so sánh Accuracy, F1-Score, ROC-AUC.

## 5. Phân tích Dữ liệu (EDA)

Bạn có thể chạy script EDA riêng biệt để tạo các báo cáo đồ thị mới nhất:

```bash
python src/eda.py
```

Các biểu đồ sẽ được lưu vào `reports/figures/`.
