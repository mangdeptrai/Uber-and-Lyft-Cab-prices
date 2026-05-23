import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("--- 1. Bắt đầu đọc và tiền xử lý dữ liệu ---")
df = pd.read_csv('uber_lyft_clean_ready.csv')
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Mã hóa dữ liệu chữ thành số và lưu bảng mapping
mapping_dict = {}
categorical_cols = ['cab_type', 'ride_type', 'source', 'destination']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    mapping_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Xuất file mapping quan trọng cho Streamlit
joblib.dump(mapping_dict, 'label_mappings.pkl')
print("Đã xử lý xong các biến phân loại.")

# 2. CHIA TẬP TRAIN/TEST (80% / 20%)
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
print("\n--- 2. Đang huấn luyện mô hình tối ưu Random Forest ---")
start_time = time.time()
# Tối ưu số cây và độ sâu để máy chạy siêu nhanh không bị treo
best_model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
print(f"Hoàn thành huấn luyện trong: {round(time.time() - start_time, 2)} giây.")

# 4. ĐÁNH GIÁ MÔ HÌNH 
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n" + "="*15 + " KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH " + "="*15)
print(f"- Hệ số xác định R2 Score: {round(r2, 4)}")
print(f"- Sai số tuyệt đối trung bình MAE: {round(mae, 4)} USD")
print(f"- Sai số căn phương trung bình RMSE: {round(rmse, 4)} USD")
print("="*56)
joblib.dump(best_model, 'uber_lyft_rf_model.pkl')
print("\nĐã lưu file 'uber_lyft_rf_model.pkl' thành công!")

# 5. XUẤT ĐỒ THỊ FEATURE IMPORTANCE 
print("\n--- 3. Đang vẽ biểu đồ độ quan trọng các đặc trưng ---")
importances = best_model.feature_importances_
feature_names = X.columns
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 8)) 
sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis', edgecolor='black')
plt.xscale('log')

plt.title('Feature Importance (Log Scale) - Các yếu tố quyết định giá cước', fontsize=14, fontweight='bold')
plt.xlabel('Mức độ đóng góp vào mô hình (Thang đo Logarit - Phóng to giá trị nhỏ)', fontsize=11)
plt.ylabel('Các thuộc tính dữ liệu', fontsize=11)

plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('feature_importance_clear.png')
print("Đã lưu ảnh biểu đồ rõ nét thành file 'feature_importance.png'")
plt.show()