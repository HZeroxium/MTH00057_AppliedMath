import cmath
import numpy as np  # type: ignore
from matrix import (
    Matrix,
    get_identity_matrix,
    back_substitution,
    inverse,
    Gauss_elimination,
)


def solve_3rd_degree_polynomial(a, b, c, d) -> list:
    if a == 0:
        raise ValueError(
            "Coefficient 'a' must not be zero for a 3rd degree polynomial."
        )

    # Chuyển đổi phương trình thành dạng: t^3 + pt + q = 0
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    # Sử dụng công thức Cardano
    discriminant = ((q**2) / 4) + ((p**3) / 27)
    # Round to 0 if very close to 0
    if abs(discriminant) < 1e-10:
        discriminant = 0
    if discriminant > 0:
        u = (-q / 2 + cmath.sqrt(discriminant)) ** (1 / 3)
        v = (-q / 2 - cmath.sqrt(discriminant)) ** (1 / 3)
        roots = [u + v - (b / (3 * a))]
    elif discriminant == 0:
        u = 0
        if q > 0:
            u = -((q / 2) ** (1.0 / 3))
        else:
            u = (-q / 2) ** (1.0 / 3)
        roots = [2 * u - (b / (3 * a)), -u - (b / (3 * a))]
    else:
        rho = cmath.sqrt(-(p**3) / 27)
        theta = cmath.acos(-q / (2 * rho))
        u = 2 * rho ** (1 / 3)
        roots = [
            u * cmath.cos(theta / 3) - (b / (3 * a)),
            u * cmath.cos((theta + 2 * cmath.pi) / 3) - (b / (3 * a)),
            u * cmath.cos((theta + 4 * cmath.pi) / 3) - (b / (3 * a)),
        ]

    # Chuyển đổi kết quả thành số thực nếu phần ảo bằng 0
    real_roots = [root.real if abs(root.imag) < 1e-10 else root for root in roots]
    return real_roots


def solve_2nd_degree_polynomial(a, b, c) -> list:
    if a == 0:
        raise ValueError(
            "Coefficient 'a' must not be zero for a 2nd degree polynomial."
        )

    discriminant = b**2 - 4 * a * c

    if discriminant > 0:
        roots = [
            (-b + discriminant**0.5) / (2 * a),
            (-b - discriminant**0.5) / (2 * a),
        ]
    elif discriminant == 0:
        roots = [-b / (2 * a)]
    else:
        roots = [
            (-b + discriminant**0.5 * 1j) / (2 * a),
            (-b - discriminant**0.5 * 1j) / (2 * a),
        ]

    # Chuyển đổi kết quả thành số thực nếu phần ảo bằng 0
    real_roots = [root.real if abs(root.imag) < 1e-10 else root for root in roots]
    return real_roots


def get_characteristic_polynomial(matrix: Matrix) -> list:
    if not matrix.is_square():
        raise ValueError("Matrix must be square to have a characteristic polynomial.")

    n = matrix.rows
    if n == 1:
        return [matrix[0][0]]
    elif n == 2:
        return [1, -matrix.trace(), matrix.getDeterminant()]
    elif n == 3:
        matrix_square = matrix * matrix
        a = 1
        b = -matrix.trace()
        c = (matrix.trace() ** 2 - matrix_square.trace()) / 2
        d = -matrix.getDeterminant()
        return [a, b, c, d]
    else:
        raise ValueError(
            "Characteristic polynomial is not implemented for matrices larger than 3x3."
        )


def find_eigenvalues(matrix: Matrix) -> list:
    if not matrix.is_square():
        raise ValueError("Matrix must be square to have a characteristic polynomial.")

    n = matrix.rows
    if n == 1:
        return [matrix[0][0]]
    elif n == 2:
        return solve_2nd_degree_polynomial(1, -matrix.trace(), matrix.getDeterminant())
    elif n == 3:
        a, b, c, d = get_characteristic_polynomial(matrix)
        return solve_3rd_degree_polynomial(a, b, c, d)
    else:
        raise ValueError(
            "Characteristic polynomial is not implemented for matrices larger than 3x3."
        )


def format_solution(roots: list) -> list:
    # Remove duplicate roots
    roots = list(set(roots))
    # If roots include complex numbers, return
    if any(isinstance(root, complex) for root in roots):
        return roots
    # Sort roots in ascending order
    roots.sort()
    # If a value is very close to an integer, round it to the nearest integer
    roots = [round(root) if abs(root - round(root)) < 1e-10 else root for root in roots]
    return roots


def find_eigenvectors(matrix: Matrix, eigenvalues: list) -> list:
    eigenvectors: list[list] = []
    for eigenvalue in eigenvalues:
        # print("Với trị riêng:", eigenvalue)
        temp: Matrix = matrix - get_identity_matrix(matrix.rows) * eigenvalue
        temp = temp.addColumn([0 for _ in range(temp.rows)])
        # print("Matrix A - λI:")
        # print(temp)
        temp = Gauss_elimination(temp)
        # print(temp)
        solutions: list[list] = back_substitution(temp)
        # print("Solutions:", solutions)
        if len(solutions) > 1:
            # Xóa các nghiệm không phải nghiệm tự do
            if solutions[0] == [0 for _ in range(len(solutions[0]))]:
                solutions.pop(0)
            for vector in solutions:
                eigenvectors.append(vector)
    return eigenvectors


def diagonalize_matrix(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix, bool]:
    eigenvalues = format_solution(find_eigenvalues(matrix))
    # print("Trị riêng:", eigenvalues)
    eigenvectors = find_eigenvectors(matrix, eigenvalues)
    # print("Vector riêng:", eigenvectors)
    if len(eigenvectors) != matrix.rows:
        return None, None, None, False
    matrix_P = Matrix(eigenvectors).transpose()
    # print(matrix_P)
    matrix_D = inverse(matrix_P) * matrix * matrix_P
    return matrix_P, matrix_D, inverse(matrix_P), True


def main():
    matrices: list[Matrix] = [
        Matrix([[-1, 3], [-2, 4]]),
        Matrix([[5, 2], [9, 2]]),
        Matrix([[1, -1, -1], [1, 3, 1], [-3, 1, -1]]),
        Matrix([[5, -1, 1], [-1, 2, -2], [1, -2, 2]]),
        Matrix([[1, 3, 3], [-3, -5, -3], [3, 3, 1]]),
        Matrix([[4, 0, -1], [0, 3, 0], [1, 0, 2]]),
        Matrix([[3, 4, -4], [-2, -1, 2], [-2, 0, 1]]),
        Matrix([[0, 0, -2], [1, 2, 1], [1, 0, 3]]),
        Matrix([[1, 0, 0], [1, 2, 0], [-3, 5, 2]]),
        Matrix([[4, 0, 1], [-2, 1, 0], [-2, 0, 1]]),
    ]

    n = len(matrices)

    for i in range(n):
        print(f"*** Ma trận câu", chr(ord("a") + i), ":")
        print(matrices[i])
        matrix_P, matrix_D, matrix_P_inv, is_diagonalizable = diagonalize_matrix(
            matrices[i]
        )
        if not is_diagonalizable:
            print("Không thể chéo hóa ma trận.")
            print("=" * 50)
            continue
        print("--> Matrix P:")
        print(matrix_P)
        print("--> Matrix P^-1:")
        print(matrix_P_inv)
        print("--> Matrix D:")
        print(matrix_D)
        print("=" * 50)

    matrices_np: list[np.ndarray] = [
        np.array([[-1, 3], [-2, 4]]),
        np.array([[5, 2], [9, 2]]),
        np.array([[1, -1, -1], [1, 3, 1], [-3, 1, -1]]),
        np.array([[5, -1, 1], [-1, 2, -2], [1, -2, 2]]),
        np.array([[1, 3, 3], [-3, -5, -3], [3, 3, 1]]),
        np.array([[4, 0, -1], [0, 3, 0], [1, 0, 2]]),
        np.array([[3, 4, -4], [-2, -1, 2], [-2, 0, 1]]),
        np.array([[0, 0, -2], [1, 2, 1], [1, 0, 3]]),
        np.array([[1, 0, 0], [1, 2, 0], [-3, 5, 2]]),
        np.array([[4, 0, 1], [-2, 1, 0], [-2, 0, 1]]),
    ]

    n = len(matrices_np)

    for i in range(n):
        print(f"*** Ma trận câu", chr(ord("a") + i), ":")
        print(matrices_np[i])
        eigenvalues, eigenvectors = np.linalg.eig(matrices_np[i])
        print("--> Trị riêng:")
        print(eigenvalues)
        print("--> Vector riêng:")
        print(eigenvectors)

        D = np.diag(eigenvalues)
        P = eigenvectors
        P_inv = np.linalg.inv(P)
        print("--> Ma trận P:")
        print(P)
        print("--> Ma trận P^-1:")
        print(P_inv)
        print("--> Ma trận D:")
        print(D)
        print("=" * 50)


if __name__ == "__main__":
    main()
