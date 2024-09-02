import numpy as np  # type: ignore
import sympy as sp  # type: ignore

# Ma trận A ban đầu
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

for i in range(len(matrices_np)):
    print(
        "============================================================================================"
    )
    print(f"*** Câu", chr(ord("a") + i), ":")
    A = matrices_np[i]

    # Kích thước của ma trận
    n = A.shape[0]

    # Tạo ma trận đơn vị
    I = np.eye(n)

    # Ký hiệu lambda
    lambda_sym = sp.symbols("λ")

    # Ma trận (A - I*lambda)
    A_lambda_I = A - I * lambda_sym
    print("---> B1: Ma trận A - Iλ:")
    print(A_lambda_I)

    # Chuyển đổi ma trận (A - I*lambda) sang dạng SymPy Matrix
    A_lambda_I_sym = sp.Matrix(A_lambda_I)

    # Tìm định thức của ma trận (A - I*lambda)
    char_poly = A_lambda_I_sym.det()
    print("---> B2: Đa thức đặc trưng của ma trận:")
    print(char_poly)

    # Giải đa thức đặc trưng để tìm các trị riêng (eigenvalues)
    eigenvalues = sp.solve(char_poly, lambda_sym)
    print("---> B3: Các trị riêng (eigenvalues):")
    print(eigenvalues)

    # Tìm các eigenvectors tương ứng với mỗi trị riêng
    eigenvectors = []
    for val in eigenvalues:
        eig_matrix = A_lambda_I_sym.subs(lambda_sym, val)
        eig_vectors = eig_matrix.nullspace()
        eigenvectors.append(eig_vectors)

    print("---> B4: Các vector riêng (eigenvectors):")
    for i, vectors in enumerate(eigenvectors):
        print(f"===> Trị riêng {eigenvalues[i]}: {vectors}")

    # Kiểm tra xem có thể chéo hóa ma trận không
    if len(eigenvectors) != n:
        print("!!! Không thể chéo hóa ma trận.")
        continue

    # Tạo ma trận P từ các eigenvectors
    P = sp.Matrix.hstack(*[vec[0] for vec in eigenvectors])

    # Ma trận đường chéo D từ các eigenvalues
    D = sp.diag(*eigenvalues)

    print("===> Ma trận P (các vector riêng):")
    print(P)
    print("===> Ma trận đường chéo D:")
    print(D)

    # Kiểm tra kết quả bằng cách tính ngược lại A = P * D * P^(-1)
    A_reconstructed = P * D * P.inv()
    print("===> Kiểm tra lại ma trận A:")
    print(A_reconstructed)
