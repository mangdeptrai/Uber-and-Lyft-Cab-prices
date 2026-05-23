import joblib

print("--- Đang tải bộ não AI từ file .pkl lên... ---")
# Nạp mô hình vào lại bộ nhớ Python
model = joblib.load('uber_lyft_rf_model.pkl')

print("\n" + "="*20 + " THÔNG TIN MÔ HÌNH ĐÃ LƯU " + "="*20)
print(f"- Loại mô hình: {type(model).__name__}")
print(f"- Số lượng cây quyết định bên trong rừng: {len(model.estimators_)} cây")
print(f"- Độ sâu tối đa của các cây (max_depth): {model.max_depth}")
print(f"- Số lượng đặc trưng đầu vào mà mô hình nhận diện: {model.n_features_in_}")
print("="*66)

print("\nCác tham số cấu hình chi tiết của thuật toán:")
print(model.get_params())