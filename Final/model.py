import math
from matrix import Matrix, inverse


class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X: Matrix, y: Matrix):
        """
        Huấn luyện mô hình hồi quy tuyến tính bằng phương pháp bình phương tối thiểu.

        Args:
            X (Matrix): Ma trận đầu vào (mẫu huấn luyện).
            y (Matrix): Ma trận đầu ra (mẫu huấn luyện).
        """
        # Thêm một cột chứa giá trị 1 vào ma trận X để tính hệ số chặn
        ones = Matrix([[1] for _ in range(X.rows)])
        X = ones.attach_matrix_horizontal(X)

        # Tính toán hệ số hồi quy bằng công thức: β = (X^T * X)^-1 * X^T * y
        X_transpose = X.transpose()
        XTX = X_transpose * X
        XTX_inv = inverse(XTX)
        XTy = X_transpose * y
        self.coefficients = XTX_inv * XTy

    def predict(self, X: Matrix) -> Matrix:
        """
        Dự đoán giá trị đầu ra cho mẫu mới dựa trên mô hình đã huấn luyện.

        Args:
            X (Matrix): Ma trận đầu vào (mẫu mới).

        Returns:
            Matrix: Giá trị dự đoán.
        """
        if self.coefficients is None:
            raise ValueError("Model has not been trained yet.")

        # Thêm một cột chứa giá trị 1 vào ma trận X để tính hệ số chặn
        ones = Matrix([[1] for _ in range(X.rows)])
        X = ones.attach_matrix_horizontal(X)

        # Dự đoán giá trị y = X * β
        predictions = X * self.coefficients
        return predictions

    def mean_squared_error(self, y_true: Matrix, y_pred: Matrix) -> float:
        """
        Tính toán lỗi trung bình bình phương (Mean Squared Error).

        Args:
            y_true (Matrix): Giá trị thực tế.
            y_pred (Matrix): Giá trị dự đoán.

        Returns:
            float: Lỗi trung bình bình phương.
        """
        errors = y_true - y_pred
        squared_errors = errors.transpose() * errors
        mse = squared_errors[0][0] / len(y_true.data)

        # Nếu giá trị MSE là một số vô cực thì trả về giá trị lớn nhất của float
        if math.isinf(mse):
            print("MSE is infinite")
            return float("inf")
        return mse

    # Calculate RSS (Residual Sum of Squares)
    def residual_sum_of_squares(self, y_true: Matrix, y_pred: Matrix) -> float:
        """
        Tính toán tổng bình phương sai số (Residual Sum of Squares).

        Args:
            y_true (Matrix): Giá trị thực tế.
            y_pred (Matrix): Giá trị dự đoán.

        Returns:
            float: Tổng bình phương sai số.
        """
        errors = y_true - y_pred
        rss = (errors.transpose() * errors)[0][0]
        return rss


class PolynomialRegression(LinearRegression):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def fit(self, X: Matrix, y: Matrix):
        # Tạo ra ma trận đặc trưng mở rộng với các bậc cao hơn
        X_poly = self._expand_features(X)
        super().fit(X_poly, y)

    def predict(self, X: Matrix) -> Matrix:
        X_poly = self._expand_features(X)
        return super().predict(X_poly)

    def _expand_features(self, X: Matrix) -> Matrix:
        expanded_data = []
        for row in X.data:
            new_row = []
            for val in row:
                new_row.extend([val**i for i in range(1, self.degree + 1)])
            expanded_data.append(new_row)
        return Matrix(expanded_data)


class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, X: Matrix, y: Matrix):
        ones = Matrix([[1] for _ in range(X.rows)])
        X = ones.attach_matrix_horizontal(X)
        X_transpose = X.transpose()
        XTX = X_transpose * X
        regularization_matrix = Matrix(
            [
                [self.alpha if i == j else 0 for j in range(XTX.columns)]
                for i in range(XTX.rows)
            ]
        )
        XTX_inv = (XTX + regularization_matrix).getInverse()
        XTy = X_transpose * y
        self.coefficients = XTX_inv * XTy


class LassoRegression(LinearRegression):
    def __init__(self, alpha=1.0, max_iter=1000, learning_rate=0.01):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X: Matrix, y: Matrix):
        """
        Huấn luyện mô hình hồi quy Lasso bằng phương pháp gradient descent.

        Args:
            X (Matrix): Ma trận đầu vào (mẫu huấn luyện).
            y (Matrix): Ma trận đầu ra (mẫu huấn luyện).
        """
        ones = Matrix([[1] for _ in range(X.rows)])
        X = ones.attach_matrix_horizontal(X)
        m, n = X.rows, X.columns
        self.coefficients = Matrix([[0.0] for _ in range(n)])

        for _ in range(self.max_iter):
            print("Running iteration", _)
            for j in range(n):
                residual = y - (X * self.coefficients)
                gradient = (X.transpose() * residual)[j][0]

                # Cập nhật hệ số với gradient descent và L1 regularization
                self.coefficients[j][0] = (
                    self.coefficients[j][0] + self.learning_rate * gradient
                )

                # Áp dụng L1 regularization
                if self.coefficients[j][0] > 0:
                    self.coefficients[j][0] = max(
                        0, self.coefficients[j][0] - self.alpha * self.learning_rate
                    )
                else:
                    self.coefficients[j][0] = min(
                        0, self.coefficients[j][0] + self.alpha * self.learning_rate
                    )


class ElasticNetRegression(LinearRegression):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, learning_rate=0.01):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X: Matrix, y: Matrix):
        ones = Matrix([[1] for _ in range(X.rows)])
        X = ones.attach_matrix_horizontal(X)
        m, n = X.rows, X.columns
        self.coefficients = Matrix([[0] for _ in range(n)])

        for _ in range(self.max_iter):
            # print("Running iteration", _)
            for j in range(n):
                residual = y - (X * self.coefficients)
                gradient = (X.transpose() * residual)[j][0]  # Lấy giá trị số thực

                l1_term = (
                    self.l1_ratio
                    * self.alpha
                    * math.copysign(1, self.coefficients[j][0])
                )
                l2_term = (1 - self.l1_ratio) * self.alpha * self.coefficients[j][0]

                # Cập nhật hệ số với gradient descent và Elastic Net penalty
                self.coefficients[j][0] = self.coefficients[j][
                    0
                ] + self.learning_rate * (gradient - l1_term - l2_term)


class BayesianLinearRegression(LinearRegression):
    def __init__(self, prior_mean=0, prior_variance=1):
        super().__init__()
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

    def fit(self, X: Matrix, y: Matrix):
        ones = Matrix([[1] for _ in range(X.rows)])
        X = ones.attach_matrix_horizontal(X)
        X_transpose = X.transpose()
        XTX = X_transpose * X
        prior_matrix = Matrix(
            [
                [self.prior_variance if i == j else 0 for j in range(XTX.columns)]
                for i in range(XTX.rows)
            ]
        )
        XTX_inv = (XTX + prior_matrix).getInverse()
        XTy = X_transpose * y + Matrix([[self.prior_mean] for _ in range(XTX.columns)])
        self.coefficients = XTX_inv * XTy


class InteractionTermsRegression(LinearRegression):
    def __init__(self):
        super().__init__()

    def add_interactions(self, X: Matrix):
        new_columns = []
        for i in range(X.columns):
            for j in range(i + 1, X.columns):
                interaction_column = [X[k][i] * X[k][j] for k in range(X.rows)]
                new_columns.append(interaction_column)

        for column in new_columns:
            X = X.addColumn(column)

        return X

    def fit(self, X: Matrix, y: Matrix):
        X = self.add_interactions(X)
        super().fit(X, y)

    def predict(self, X: Matrix) -> Matrix:
        X = self.add_interactions(X)
        return super().predict(X)
