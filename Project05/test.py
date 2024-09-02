# Topic: File "wine.csv" is a database that evaluates the quality of 1,200 bottles of wine on a scale of 1-10 based on 11 different properties.
# Example wine.csv
# "fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";" alcohol";"quality"
# 7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
# 7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5
# 7.8;0.76;0.04;2.3;0.092;15;54;0.997;3.26;0.65;9.8;5
# 11.2;0.28;0.56;1.9;0.075;17;60;0.998;3.16;0.58;9.8;6
# 7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
# ...
# ...


# Build a model to evaluate wine quality using linear regression method.
# a. Use all 11 topic features provided.
# b. Use only 1 feature for best results.
# c. Build your own model for best results.

# Note: The source code must not use specialized libraries for linear regression such as sklearn. Only use libraries such as pandas to read data, numpy, sympy to calculate and matplotlib to draw graphs. During the process of writing code, you need to carefully annotate the meaning of all functions in the library. You can write additional functions to avoid repeating calculation operations.
import itertools
import pandas as pd  # type: ignore: Library to support data processing
import numpy as np  # type: ignore: Library to support matrix calculation
import matplotlib.pyplot as plt  # type: ignore: Library to support drawing graphs

# Read data from wine.csv
data = pd.read_csv("wine.csv", delimiter=";")

# Split data into input and output
X = data.iloc[:, :-1].values  # 11 feature labels of wine
y = data.iloc[:, -1].values  # quality of wine
print(X)


# Hàm tính toán hệ số của phương trình hồi quy tuyến tính
def linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Thêm cột 1 vào ma trận X
    X = np.insert(X, 0, 1, axis=1)
    # Tính ma trận nghịch đảo của X.T * X
    XTX_inv = np.linalg.inv(X.T @ X)
    # Tính hệ số w = (X.T * X)^-1 * X.T * y
    w = XTX_inv @ X.T @ y
    return w


# Hàm dự đoán giá trị dựa trên hệ số w và giá trị đầu vào X
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Thêm cột 1 vào ma trận X
    X = np.insert(X, 0, 1, axis=1)
    # Dự đoán giá trị dựa trên hệ số w và giá trị đầu vào X
    y_pred = X @ w
    return y_pred


# Hàm tính toán lỗi bình phương trung bình (MSE)
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Tính lỗi bình phương trung bình
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


# Hàm vẽ biểu đồ dữ liệu
def plot_data(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
    plt.scatter(X, y, color="blue", label="Actual")  # Vẽ dữ liệu thực tế
    plt.plot(
        X, y_pred, color="red", linewidth=2, label="Predicted"
    )  # Vẽ dữ liệu dự đoán
    plt.xlabel("Feature")  # Đặt tên cho trục x
    plt.ylabel("Quality")  # Đặt tên cho trục y
    plt.title("Linear Regression")  # Đặt tiêu đề cho biểu đồ
    plt.legend()  # Hiển thị chú thích
    plt.show()  # Hiển thị biểu đồ


# Hàm sử dụng toàn bộ 11 đặc trưng để dự đoán chất lượng rượu
def all_features() -> None:
    # Sử dụng toàn bộ 11 đặc trưng
    X_all = X
    # Tính hệ số w
    w_all = linear_regression(X_all, y)
    # Dự đoán chất lượng rượu
    y_pred_all = predict(X_all, w_all)
    # Tính lỗi bình phương trung bình
    mse_all = mean_squared_error(y, y_pred_all)
    print(f"Mean Squared Error (All Features): {mse_all}")
    # Vẽ biểu đồ dữ liệu

    plot_data(X[:, 0], y, y_pred_all)

    # What is X[:, 0]? Answer: X[:, 0] is the first column of the X matrix. In this case, it is the "fixed acidity" feature of the wine dataset.


# Hàm sử dụng 1 đặc trưng tốt nhất để dự đoán chất lượng rượu
def best_feature() -> None:
    best_mse = float("inf")
    best_feature = None
    best_w = None
    for i in range(X.shape[1]):
        # Sử dụng 1 đặc trưng
        X_single_feature = X[:, i].reshape(-1, 1)
        # Tính hệ số w
        w_single = linear_regression(X_single_feature, y)
        # Dự đoán chất lượng rượu
        y_pred_single = predict(X_single_feature, w_single)
        # Tính lỗi bình phương trung bình
        mse_single = mean_squared_error(y, y_pred_single)
        print(f"Feature: {i}")  # In ra đặc trưng đang xét
        print(f"Mean Squared Error (Feature {i}): {mse_single}")
        if mse_single < best_mse:
            best_mse = mse_single
            best_feature = i
            best_w = w_single

    print(f"Best Feature: {best_feature}")
    print(f"Mean Squared Error (Best Feature): {best_mse}")
    # Vẽ biểu đồ dữ liệu
    plot_data(X[:, best_feature], y, predict(X[:, best_feature].reshape(-1, 1), best_w))


# c. Xây dựng mô hình của riêng bạn cho kết quả tốt nhất
def best_model():
    best_mse = float("inf")
    best_combination = None
    best_beta = None

    # Tìm tất cả các tổ hợp đặc trưng
    with open("output.txt", "w") as f:
        for k in range(1, X.shape[1] + 1):
            for subset in itertools.combinations(range(X.shape[1]), k):
                # Sử dụng tổ hợp đặc trưng
                X_subset = X[:, list(subset)]
                # Tính hệ số beta
                beta_subset = linear_regression(X_subset, y)
                # Dự đoán chất lượng rượu
                y_pred_subset = predict(X_subset, beta_subset)
                # Tính lỗi bình phương trung bình
                mse_subset = mean_squared_error(y, y_pred_subset)
                f.write(f"Combination: {subset}\n")
                f.write(f"Mean Squared Error (Combination {subset}): {mse_subset}\n")
                if mse_subset < best_mse:
                    best_mse = mse_subset
                    best_combination = subset
                    best_beta = beta_subset

    print(f"Best Combination: {best_combination}")
    print(f"Mean Squared Error (Best Combination): {best_mse}")

    # Vẽ biểu đồ dữ liệu
    plot_data(
        X[:, best_combination],
        y,
        predict(X[:, best_combination], best_beta),
    )


# Hàm tạo đặc trưng đa thức
def polynomial_features(X, degree):
    poly_X = X.copy()
    for d in range(2, degree + 1):
        poly_X = np.hstack((poly_X, X**d))
    return poly_X


# c. Xây dựng mô hình đa thức của riêng bạn cho kết quả tốt nhất
def best_polynomial_model(max_degree):
    best_mse = float("inf")
    best_combination = None
    best_degree = 0
    best_beta = None

    # Tìm tất cả các tổ hợp đặc trưng
    with open("output.txt", "w") as f:
        for k in range(1, X.shape[1] + 1):
            for subset in itertools.combinations(range(X.shape[1]), k):
                for degree in range(1, max_degree + 1):
                    # Sử dụng tổ hợp đặc trưng và tạo đặc trưng đa thức
                    X_subset = X[:, list(subset)]
                    X_poly = polynomial_features(X_subset, degree)
                    # Tính hệ số beta
                    beta_subset = linear_regression(X_poly, y)
                    # Dự đoán chất lượng rượu
                    y_pred_subset = predict(X_poly, beta_subset)
                    # Tính lỗi bình phương trung bình
                    mse_subset = mean_squared_error(y, y_pred_subset)
                    f.write(f"Combination: {subset}, Degree: {degree}\n")
                    f.write(
                        f"Mean Squared Error (Combination {subset}, Degree {degree}): {mse_subset}\n"
                    )
                    if mse_subset < best_mse:
                        best_mse = mse_subset
                        best_combination = subset
                        best_degree = degree
                        best_beta = beta_subset

    print(f"Best Combination: {best_combination}")
    print(f"Best Degree: {best_degree}")
    print(f"Mean Squared Error (Best Combination): {best_mse}")

    # Vẽ biểu đồ dữ liệu
    X_best = X[:, best_combination]
    X_best_poly = polynomial_features(X_best, best_degree)
    y_pred_best = predict(X_best_poly, best_beta)

    plot_data(X_best, y, y_pred_best)


def main() -> None:
    # all_features()
    # best_feature()
    # best_model()
    best_polynomial_model(3)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Đọc dữ liệu từ file CSV
    data = pd.read_csv("wine.csv")

    # Kiểm tra các cột trong dữ liệu
    print(data.columns)

    # Tách biến đầu vào (X) và biến đầu ra (y)
    X = data.iloc[:, :-1].values  # 11 feature labels of wine
    y = data.iloc[:, -1].values  # quality of wine

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Khởi tạo mô hình hồi quy tuyến tính
    model = LinearRegression()

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")

    # Trích xuất các hệ số hồi quy
    coefficients = model.coef_
    features = X.columns

    # Tạo biểu đồ trọng số của các hệ số hồi quy
    plt.figure(figsize=(10, 6))
    plt.barh(features, coefficients)
    plt.xlabel("Trọng số")
    plt.ylabel("Đặc trưng")
    plt.title("Biểu đồ trọng số của các hệ số hồi quy")
    plt.show()
