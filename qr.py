from vector import Vector, gram_schmidt
from matrix import Matrix, get_identity_matrix
import numpy as np  # type: ignore


def qr_decomposition(matrix: Matrix) -> tuple[Matrix, Matrix]:
    vectors_u = [Vector(v) for v in matrix.transpose().data]
    vectors_v = gram_schmidt(vectors_u)
    vectors_q = [v.unitize() for v in vectors_v]
    matrix_q = Matrix([v.data for v in vectors_q]).transpose()
    matrix_r = Matrix([[0] * matrix.columns for _ in range(matrix.columns)])
    for i in range(matrix.columns):
        for j in range(i, matrix.columns):
            matrix_r.data[i][j] = vectors_q[i].calDotProduct(vectors_u[j])
    return matrix_q, matrix_r


def main():

    # matrices: list[Matrix] = [
    #     Matrix([[1, 1, 2], [2, -1, 1], [-2, 4, 1]]),
    #     Matrix([[1, 1, 1], [2, -2, 2], [1, 1, -1]]),
    #     Matrix([[1, 1, -1], [0, 1, 2], [1, 1, 1]]),
    #     Matrix([[-1, -1, 1], [1, 3, 3], [-1, -1, 5], [1, 3, 7]]),
    #     Matrix([[1, 1, 1], [2, 2, 0], [3, 0, 0], [0, 0, 1]]),
    #     Matrix([[-2, 1, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #     Matrix([[1, -1, 2], [1, 0, -1], [-1, 1, 2], [0, 1, 1]]),
    # ]

    matrices_np = [
        np.array([[1, 1, 2], [2, -1, 1], [-2, 4, 1]]),
        np.array([[1, 1, 1], [2, -2, 2], [1, 1, -1]]),
        np.array([[1, 1, -1], [0, 1, 2], [1, 1, 1]]),
        np.array([[-1, -1, 1], [1, 3, 3], [-1, -1, 5], [1, 3, 7]]),
        np.array([[1, 1, 1], [2, 2, 0], [3, 0, 0], [0, 0, 1]]),
        np.array([[-2, 1, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, -1, 2], [1, 0, -1], [-1, 1, 2], [0, 1, 1]]),
    ]

    # n = len(matrices)
    # for i in range(n):
    #     print("Ma trận câu " + str(i + 1) + ":")
    #     print(matrices[i])
    #     matrix_q, matrix_r = qr_decomposition(matrices[i])
    #     print("Ma trận Q:")
    #     print(matrix_q)
    #     print("Ma trận R:")
    #     print(matrix_r)
    #     print("Ma trận Q * R:")
    #     qr: Matrix = matrix_q * matrix_r
    #     print(qr)
    #     print("=" * 50)

    n = len(matrices_np)
    for i in range(n):
        print("Ma trận câu " + chr(ord("a") + i) + ":")
        print(matrices_np[i])
        matrix_q_np, matrix_r_np = np.linalg.qr(matrices_np[i])
        print("Ma trận Q:")
        print(matrix_q_np)
        print("Ma trận R:")
        print(matrix_r_np)
        print("Ma trận Q * R:")
        qr_np = np.dot(matrix_q_np, matrix_r_np)
        print(qr_np)
        print("=" * 50)


if __name__ == "__main__":
    main()
