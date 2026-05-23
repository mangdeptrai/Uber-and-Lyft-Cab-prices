import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("KHỞI ĐỘNG HỒI QUY TUYẾN TÍNH BAN ĐẦU")
df = pd.read_csv('uber_lyft_clean_ready.csv')
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Mã hóa Label Encoding cho dữ liệu chữ
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['price'])
y = df['price']

# 2. CHIA TẬP DỮ LIỆU 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. HUẤN LUYỆN MÔ HÌNH TOÀN DIỆN
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nKết quả mô hình tổng thể: R2 = {round(r2, 4)} | RMSE = {round(rmse, 4)}")


# 4. TIẾN TRÌNH VẼ ĐỒ THỊ ĐƯỜNG THẲNG HỒI QUY THEO KHOẢNG CÁCH
plt.figure(figsize=(9, 6))

# Lấy ngẫu nhiên một tập mẫu 300 dòng để vẽ chấm dữ liệu (tránh bị đen đặc biểu đồ)
sample_df = df.sample(n=300, random_state=42)
X_sample_dist = sample_df[['distance']]  # Trục X: Khoảng cách thô
y_sample_price = sample_df['price']       # Trục Y: Giá tiền thật

# Vẽ các chấm tròn đại diện cho dữ liệu thực tế
plt.scatter(X_sample_dist, y_sample_price, color='#1f77b4', alpha=0.7, edgecolors='k', label='Dữ liệu chuyến xe thực tế')

# Huấn luyện một đường thẳng hồi quy riêng cho biến khoảng cách để vẽ đường xu hướng
lr_single = LinearRegression()
lr_single.fit(X_sample_dist, y_sample_price)

# Tạo mảng khoảng cách từ thấp đến cao để dựng đường thẳng tuyến tính
x_line = np.linspace(X_sample_dist.min(), X_sample_dist.max(), 100).reshape(-1, 1)
y_line_pred = lr_single.predict(x_line)

# Vẽ ĐƯỜNG THẲNG HỒI QUY MÀU ĐỎ cắt qua các điểm dữ liệu
plt.plot(x_line, y_line_pred, color='red', linewidth=3, label='Đường thẳng Hồi quy (Regression Line)')

# Cấu hình các thông số nhãn hiển thị y hệt mẫu nghiên cứu khoa học
plt.title('MÔ HÌNH HỒI QUY TUYẾN TÍNH\nSự phụ thuộc của Giá cước vào Khoảng cách di chuyển', fontsize=12, fontweight='bold')
plt.xlabel('Khoảng cách hành trình (Distance - Chuẩn hóa)', fontsize=10)
plt.ylabel('Giá cước chuyến đi (Price - USD)', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# Lưu đồ thị chuẩn hóa
plt.savefig('linear_regression_result.png', dpi=300)
print("Đã xuất ảnh đồ thị đường thẳng hồi quy 'linear_regression_result.png' thành công!")
plt.show()