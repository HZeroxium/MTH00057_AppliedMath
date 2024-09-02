import pandas as pd  # type: ignore - Thư viện dùng để xử lý dữ liệu dạng bảng (dataframe)
import numpy as np  # type: ignore - Thư viện dùng để xử lý dữ liệu dạng ma trận và các phép toán liên quan
import matplotlib.pyplot as plt  # type: ignore - Thư viện dùng để vẽ biểu đồ

# Đọc dữ liệu từ file "wine.csv"

data = pd.read_csv("wine.csv", delimiter=";")

# Chia dữ liệu thành 2 phần: dữ liệu đầu vào và dữ liệu đầu ra
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

features = data.columns[:-1]  # Tên các đặc trưng

print(features)
