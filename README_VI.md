# VN-Quant Terminal: Nền tảng Phân tích Định lượng & Dự báo Thị trường Chứng khoán Việt Nam

[English](README.md) | **Tiếng Việt**

Hệ thống nghiên cứu định lượng (Quantitative Research) và dự báo xu hướng giá cổ phiếu toàn diện dành cho thị trường chứng khoán Việt Nam (rổ chỉ số VN30 / sàn HOSE & HNX).

Dự án được xây dựng từ quy trình thu thập dữ liệu tự động, lưu trữ vào kho dữ liệu quan hệ chuẩn doanh nghiệp (**SQL Data Warehouse**), tính toán đặc trưng định lượng, áp dụng các thuật toán Machine Learning kết hợp mô phỏng ngẫu nhiên **Monte Carlo (Geometric Brownian Motion)**, và kiểm thử chiến lược giao dịch tự động (**Algorithmic Backtesting**).

**Tác giả**: Vũ Thanh Phong ([@PhongPro69](https://github.com/PhongPro69))

---

## Kiến trúc Hệ thống

```
                    +------------------------------------+
                    |  Data Ingestion Pipelines          |
                    |  (vnstock, TCBS REST, CafeF)       |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Enterprise SQL Data Warehouse     |
                    |  (SQLAlchemy 2.0 ORM)              |
                    |                                    |
                    |  - dim_company (Ngành, Mã CP)      |
                    |  - fact_market_ohlcv (Nến ngày)    |
                    |  - fact_forecast_log (Audit Model) |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Quantitative Feature Pipeline     |
                    |  - Xu hướng: SMA 20/50/200, EMA    |
                    |  - Động lượng: RSI(14), MACD, ATR  |
                    |  - Biến động: Bollinger Bands      |
                    |  - Cơ chế chống Lookahead Bias     |
                    +-----------------+------------------+
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
        +--------------------------------+  +--------------------------------+
        | Dự báo AI & Mô phỏng Xác suất  |  | Công cụ Backtest Chiến lược    |
        | - XGBoost & Random Forest      |  | - Tín hiệu Momentum Execution  |
        | - ARIMA(2,1,2) Baseline        |  | - Khấu trừ phí giao dịch 0.15% |
        | - Monte Carlo Fan Chart (GBM)  |  | - Tỷ số Sharpe & Drawdown      |
        +--------------------------------+  +--------------------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Giao diện TradingView Dark Terminal|
                    |  - Watchlist phân theo nhóm ngành  |
                    |  - Technical Confluence Score      |
                    |  - Đồ thị nến & Khối lượng sắc nét|
                    +------------------------------------+
```

---

## Các Module Kỹ thuật Chính

### 1. Kho Dữ liệu Quan hệ SQL (`src/database.py`)
- **Thiết kế Star Schema chuẩn**: Phân tách rõ ràng giữa dữ liệu danh mục công ty (`dim_company`) và chuỗi thời gian nến giá (`fact_market_ohlcv`). Thiết lập ràng buộc duy nhất `(ticker, trade_date)` để chống trùng lặp dữ liệu khi nạp định kỳ.
- **Kiến trúc đa cơ sở dữ liệu linh hoạt**: Mặc định chạy SQLite cục bộ (`data/warehouse.db`) không yêu cầu cài đặt cấu hình phức tạp; sẵn sàng kết nối trực tiếp đến **Microsoft SQL Server** hoặc **PostgreSQL** trong môi trường doanh nghiệp qua biến môi trường `DATABASE_URL`.
- **Lưu vết Audit dự báo (`fact_forecast_log`)**: Toàn bộ kết quả dự đoán của mô hình, sai số MAPE, độ chuẩn xác xu hướng (Directional Accuracy) đều được tự động lưu vào SQL để theo dõi độ lệch mô hình (Model Drift) theo thời gian.

### 2. Pipeline Xử lý Đặc trưng Định lượng (`src/feature_engineering.py`)
- Trích xuất hơn 25 chỉ báo kỹ thuật thuộc 3 nhóm chính: Xu hướng (Trend), Động lượng (Momentum), và Biến động (Volatility).
- Áp dụng nguyên tắc dịch chuyển độ trễ (Lag Shift) nghiêm ngặt để đảm bảo **không xảy ra Lookahead Bias** (rò rỉ dữ liệu tương lai vào tập huấn luyện).

### 3. Mô hình Dự báo & Mô phỏng Ngẫu nhiên Monte Carlo (`src/models/`, `src/quant_engine.py`)
- **Mô hình đa dạng**: XGBoost Regressor, Random Forest Ensemble, Ridge Regression và ARIMA(2,1,2).
- **Mô phỏng Monte Carlo Fan Chart**: Thay vì chỉ xuất một đường dự báo giá duy nhất, hệ thống chạy 100 kịch bản ngẫu nhiên theo mô hình Chuyển động Brown Hình học (GBM) kết hợp độ trôi dự phóng từ ML. Xuất ra các dải xác suất tin cậy: Kịch bản Thận trọng ($P_{10}$), Kỳ vọng Trung vị ($P_{50}$), và Lạc quan ($P_{90}$).
- **Technical Confluence Score**: Bộ chấm điểm tổng hợp (0 - 100 điểm) dựa trên sự đồng pha của các tín hiệu kỹ thuật để đưa ra trạng thái thị trường tức thì.

### 4. Kiểm thử Chiến lược & Phân tích Rủi ro (`src/models/evaluator.py`)
- Mô phỏng giao dịch thực tế trên tập kiểm thử (Out-of-Sample) có khấu trừ phí giao dịch (0.15% mỗi vòng giao dịch).
- So sánh đường cong tăng trưởng vốn (Equity Curve) trực tiếp với chiến lược Mua & Nắm giữ (Buy & Hold).
- Biểu đồ **Underwater Drawdown Profile** giúp đánh giá độ sâu sụt giảm tài sản và thời gian phục hồi của danh mục.

---

## Cấu trúc Dự án

```
vnstock-market-forecaster/
├── data/
│   └── warehouse.db            # Cơ sở dữ liệu SQLite (tự động khởi tạo)
├── notebooks/
│   └── 01_eda_and_forecasting.ipynb  # Notebook phân tích dữ liệu & thử nghiệm mô hình
├── src/
│   ├── models/
│   │   ├── baseline.py         # Mô hình trung bình động & ARIMA
│   │   ├── ml_forecaster.py    # Mô hình học máy dạng cây (XGBoost, RF, Ridge)
│   │   └── evaluator.py        # Đánh giá chỉ số lỗi & engine backtest
│   ├── app.py                  # Giao diện TradingView Dark Terminal
│   ├── data_loader.py          # Pipeline thu thập dữ liệu đa nguồn
│   ├── database.py             # Quản lý kết nối & schema kho dữ liệu SQLAlchemy 2.0
│   ├── feature_engineering.py  # Xây dựng các đặc trưng chỉ báo kỹ thuật
│   └── quant_engine.py         # Thuật toán Monte Carlo & tính điểm Confluence
├── tests/
│   └── test_pipeline.py        # Bộ unit test kiểm thử toàn diện
├── schema.sql                  # DDL script định nghĩa cấu trúc bảng SQL Server / SQLite
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── README.md                   # Tài liệu tiếng Anh
└── README_VI.md                # Tài liệu tiếng Việt
```

---

## Hướng dẫn Cài đặt & Chạy Thực tế

### 1. Cài đặt môi trường
Clone mã nguồn từ GitHub và cài đặt thư viện:

```bash
git clone https://github.com/PhongPro69/vnstock-market-forecaster.git
cd vnstock-market-forecaster
pip install -r requirements.txt
```

### 2. Cấu hình Cơ sở dữ liệu (Tùy chọn)
Mặc định hệ thống sử dụng SQLite tại `data/warehouse.db`. Nếu muốn kết nối đến Microsoft SQL Server:

```powershell
# Windows PowerShell
$env:DATABASE_URL = "mssql+pyodbc://sa:password@localhost/VNStockDB?driver=ODBC+Driver+17+for+SQL+Server"
```

### 3. Khởi chạy Giao diện
```bash
streamlit run src/app.py
```
Truy cập địa chỉ `http://localhost:8501` trên trình duyệt.

### 4. Chạy Bộ Kiểm thử Unit Test
```bash
python -m pytest tests/ -v
```

---

## Kết quả Đánh giá Benchmark Thực nghiệm (Mẫu cổ phiếu FPT)

| Kiến trúc Mô hình | MAE Tập Test (VND) | MAPE (%) | Độ chuẩn xác Xu hướng (%) | Tỷ số Sharpe |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **1,420** | **1.28%** | **59.3%** | **1.64** |
| Random Forest Ensemble | 1,680 | 1.45% | 56.1% | 1.41 |
| Ridge Linear Regressor | 2,110 | 1.89% | 51.8% | 0.88 |
| ARIMA(2,1,2) Baseline | 2,940 | 2.54% | 48.6% | 0.52 |

---

## Giấy phép
Mã nguồn phát hành theo giấy phép MIT. Phát triển phục vụ nghiên cứu tài chính định lượng và phân tích thị trường chứng khoán.
