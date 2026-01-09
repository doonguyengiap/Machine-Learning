# 3A Superstore Machine Learning Project

Dự án phân tích và dự báo hành vi khách hàng sử dụng bộ dữ liệu Superstore.

## 1. Cấu trúc thư mục (Repository Structure)
```
3A_Superstore/
├── README.md                # Hướng dẫn cài đặt và chạy project
├── Nx_report.pdf            # Báo cáo project (Sinh viên tự cập nhật x)
├── data/                    # Thư mục chứa dữ liệu
│   └── processed_data.csv   # Dữ liệu đã tiền xử lý
├── src/                     # Mã nguồn Python (Pipeline chuẩn)
│   ├── preprocessing.py     # Xử lý dữ liệu
│   ├── eda.py               # Phân tích dữ liệu khám phá (EDA)
│   ├── feature_engineering.py # Tạo và chọn đặc trưng
│   ├── model_student.py     # Huấn luyện mô hình (Họ và tên SV)
│   ├── evaluation.py        # Đánh giá mô hình
│   └── main.py              # Script chính để chạy toàn bộ pipeline
├── requirements.txt         # Danh sách thư viện cần cài
└── reports/                 # Chứa các biểu đồ và kết quả output
```

## 2. Hướng dẫn cài đặt (Installation)

Yêu cầu Python 3.8+. Nên sử dụng môi trường ảo (venv).

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Linux/Mac
# .venv\Scripts\activate   # Trên Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

## 3. Cách chạy Project (Usage)

Để chạy toàn bộ quá trình từ tiền xử lý, phân tích EDA đến huấn luyện và đánh giá mô hình, hãy chạy lệnh sau:

```bash
python src/main.py
```

Kết quả sẽ được lưu trong thư mục `reports/` bao gồm các biểu đồ (figures) và file so sánh mô hình (JSON).

## 4. Các thành phần chính của Pipeline
- **Preprocessing**: Tự động nhận diện dấu phân cách, làm sạch cột số và định dạng ngày tháng.
- **EDA**: Tạo các biểu đồ phân bố lợi nhuận, tương quan đặc trưng và xu hướng thời gian.
- **Feature Engineering**: Tính toán các chỉ số RFM (Recency, Frequency, Monetary) cho từng khách hàng và tạo biến mục tiêu `IS_PROFIT`.
- **Modeling**: Huấn luyện Random Forest và Logistic Regression với cân bằng trọng số lớp (class balance).
- **Evaluation**: Đánh giá chi tiết bằng Accuracy, F1-score, ROC-AUC và Confusion Matrix.
