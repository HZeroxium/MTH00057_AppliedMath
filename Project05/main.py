import pandas as pd  # type: ignore: Library to support data processing
import numpy as np  # type: ignore: Library to support matrix calculation
import matplotlib.pyplot as plt  # type: ignore: Library to support drawing graphs
from typing import Tuple
from numpy.typing import ArrayLike
from pandas import DataFrame


# Function to read data from a CSV file
def load_data(file_path: str, feature: str, target: str) -> Tuple[ArrayLike, ArrayLike]:
    data: DataFrame = pd.read_csv(file_path, delimiter=";")
    X = data[feature].values
    y = data[target].values
    return X, y


# Function to split data into training and testing sets
def train_test_split(
    X, y, test_size=0.2, random_state=42
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    test_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# Function to add an intercept term to the input data
def add_intercept(X: ArrayLike) -> ArrayLike:
    return np.hstack((np.ones((X.shape[0], 1)), X))


# Function to compute the coefficients of the linear regression equation - Model: Y = theta0 + theta1*X
def linear_regression(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    X = add_intercept(X)
    XTX_inv = np.linalg.inv(X.T @ X)
    theta = XTX_inv @ X.T @ y
    return theta


# Function to compute the coefficients of the quadratic regression equation - Model: Y = theta0 + theta1*X + theta2*X^2
def quadratic_regression(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    X = np.hstack((X, X**2))
    X = add_intercept(X)
    XTX_inv = np.linalg.inv(X.T @ X)
    theta = XTX_inv @ X.T @ y
    return theta


# Function to compute the coefficients of the cubic regression equation - Model: Y = theta0 + theta1*X + theta2*X^2 + theta3*X^3
def cube_regression(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    X = np.hstack((X, X**2, X**3))
    X = add_intercept(X)
    XTX_inv = np.linalg.inv(X.T @ X)
    theta = XTX_inv @ X.T @ y
    return theta


# Function to compute the coefficients of the exponential regression equation - Model: Y = theta0 + theta1*exp(X)
def exponential_regression(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    X_exp = np.exp(X)
    X = add_intercept(X)
    XTX_inv = np.linalg.inv(X.T @ X)
    theta = XTX_inv @ X.T @ y
    return theta


# Function to compute the coefficients of the logarithmic regression equation - Model: Y = theta0 + theta1*ln(X)
def linear_log_regression(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    X_log = np.log(X)
    X = add_intercept(X)
    XTX_inv = np.linalg.inv(X.T @ X)
    theta = XTX_inv @ X.T @ y
    return theta


# Function to compute the coefficients of the logarithmic regression equation - Model: lnY = theta0 + theta1*X
def log_linear_regression(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    y = np.log(y)
    X = add_intercept(X)
    XTX_inv = np.linalg.inv(X.T @ X)
    theta = XTX_inv @ X.T @ y
    return theta


# Function to compute the coefficients of the logarithmic regression equation - Model: lnY = theta0 + theta1*ln(X)
def log_log_regression(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    X = np.log(X)
    y = np.log(y)
    X = add_intercept(X)
    XTX_inv = np.linalg.inv(X.T @ X)
    theta = XTX_inv @ X.T @ y
    return theta


# Function to predict the target variable using the input data and the coefficients of the regression equation
def predict(X: ArrayLike, theta: ArrayLike) -> ArrayLike:
    X = add_intercept(X)
    y_pred = X @ theta
    return y_pred


# Function to compute the mean squared error
def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return np.mean((y_true - y_pred) ** 2)


def plot_data(X: ArrayLike, y: ArrayLike, y_pred: ArrayLike) -> None:
    plt.scatter(X, y, color="blue", label="Actual")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Predicted")
    plt.xlabel("Feature")
    plt.ylabel("Quality")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()


def plot_results(
    X_test,
    y_test,
    models,
    xlabel="Actual Quality",
    ylabel="Predicted Quality",
    title="Actual vs Predicted Quality",
):
    num_models = len(models)
    num_cols = 2
    num_rows = (num_models + 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axs = axs.ravel()
    for i, (name, y_pred) in enumerate(models.items()):
        axs[i].scatter(X_test, y_test, color="blue", label="Actual Quality")
        axs[i].plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Quality")
        axs[i].set_xlabel("Feature")
        axs[i].set_ylabel("Quality")
        axs[i].set_title(name)
        axs[i].legend()
    plt.suptitle(title)
    plt.show()


def main_regression(file_path, feature, target="quality"):
    X, y = load_data(file_path, feature, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    models = {}

    # Linear Regression Y = theta_1 + theta_2 * X
    theta_linear = linear_regression(X_train, y_train)
    y_pred_linear = predict(X_test, theta_linear)
    models["Linear Regression"] = y_pred_linear

    # Quadratic Regression Y = theta_1 + theta_2 * X^2
    theta_quadratic = quadratic_regression(X_train, y_train)
    X_test_quad = np.hstack((X_test, X_test**2))
    y_pred_quadratic = predict(X_test_quad, theta_quadratic)
    models["Quadratic Regression"] = y_pred_quadratic

    # Cube Regression Y = theta_1 + theta_2 * X + theta_3 * X^2 + theta_4 * X^3
    theta_cube = cube_regression(X_train, y_train)
    X_test_cube = np.hstack((X_test, X_test**2, X_test**3))
    y_pred_cube = predict(X_test_cube, theta_cube)
    models["Cube Regression"] = y_pred_cube

    # Exponential Regression Y = theta_1 + theta_2 * e^X
    theta_exp = exponential_regression(X_train, y_train)
    X_test_exp = np.hstack((X_test, np.exp(X_test)))
    y_pred_exp = predict(X_test_exp, theta_exp)
    models["Exponential Regression"] = y_pred_exp

    # LinearLog Regression Y = theta_1 + theta_2 * ln(X)
    theta_linear_log = linear_log_regression(X_train, y_train)
    X_test_log = np.log(X_test)
    y_pred_linear_log = predict(X_test_log, theta_linear_log)
    models["LinearLog Regression"] = y_pred_linear_log

    # LogLinear Regression ln(Y) = theta_1 + theta_2 * X
    theta_log_linear = log_linear_regression(X_train, y_train)
    y_pred_log_linear = np.exp(predict(X_test, theta_log_linear))
    models["LogLinear Regression"] = y_pred_log_linear

    # LogLog Regression ln(Y) = theta_1 + theta_2 * ln(X)
    theta_log_log = log_log_regression(X_train, y_train)
    y_pred_log_log = np.exp(predict(X_test_log, theta_log_log))
    models["LogLog Regression"] = y_pred_log_log

    plot_results(X_test, y_test, models)


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


# Function to process the data and perform linear regression using all features
def all_features(X: np.ndarray, y: np.ndarray) -> None:
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


def main():
    # Read data from wine.csv
    data = pd.read_csv("wine.csv", delimiter=";")

    # Split data into input and output
    X = data.iloc[:, :-1].values  # 11 feature labels of wine
    y = data.iloc[:, -1].values  # quality of wine
    # features = data.columns[:-1]  # feature labels of wine
    # for feature in features:
    first_row = X[0]
    print(first_row)
    first_row = np.log(first_row + 1e-10)
    print(first_row)

    # all_features(X, y)


if __name__ == "__main__":
    main()
