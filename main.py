from matrix import (
    Matrix,
    Gauss_elimination,
    back_substitution,
    get_identity_matrix,
    inverse,
    pow,
)
from vector import gram_schmidt, Vector


def roundSol(solution: list, n: int) -> list:
    if len(solution) == 0:
        return solution
    if type(solution[0]) != list:
        return [round(x, n) for x in solution]
    len_ = len(solution)
    for i in range(len_):
        solution[i] = [round(x, n) for x in solution[i]]


def main():
    # matrix_a: "Matrix" = Matrix([[1, 1, 2], [2, -1, 1], [-2, 4, 1]])
    # matrix_b: "Matrix" = Matrix([[1, 1, 1], [2, -2, 2], [1, 1, -1]])
    # matrix_c: "Matrix" = Matrix([[1, 1, -1], [0, 1, 2], [1, 1, 1]])
    # matrix_d = Matrix([[-1, -1, 1], [1, 3, 3], [-1, -1, 5], [1, 3, 7]])
    # matrix_e = Matrix([[1, 1, 1], [2, 2, 0], [3, 0, 0], [0, 0, 1]])
    # matrix_f = Matrix([[-2, 1, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # matrix_g = Matrix([[1, -1, 2], [1, 0, -1], [-1, 1, 2], [0, 1, 1]])
    # matrix_h = Matrix([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]])

    # matrix_h = matrix_h.transpose()

    # vectors = [Vector(v) for v in matrix_h.transpose().data]
    # vectors = gram_schmidt(vectors)

    # print("Orthonormal basis:")
    # for v in vectors:
    #     print(v)

    # unitize_vectors = [v.unitize() for v in vectors]
    # print("\nUnit vectors:")
    # for v in unitize_vectors:
    #     print(v)

    # matrix_q = Matrix([v.data for v in unitize_vectors]).transpose()
    # print("\nMatrix Q:")
    # print(matrix_q)

    # for i in range(len(unitize_vectors)):
    #     for j in range(i, len(unitize_vectors)):
    #         temp = round(unitize_vectors[i].calDotProduct(unitize_vectors[j]), 2)
    #         print("<q" + str(i + 1) + ", u" + str(j + 1) + "> = ", end="")
    #         print(temp)

    #############################

    matrices: list[Matrix] = [
        #     Matrix([[1, 2, -1], [2, 2, 1], [3, 5, -2]]),
        #     Matrix([[1, -2, -1], [2, -3, 1], [3, -5, 0], [1, 0, 5]]),
        #     Matrix([[1, 2, 0, 2], [3, 5, -1, 6], [2, 4, 1, 2], [2, 0, -7, 11]]),
        #     Matrix([[2, -4, -1], [1, -3, 1], [3, -5, -3]]),
        #     Matrix([[1, 2, -2], [3, -1, 1], [-1, 5, -5]]),
        #     Matrix([[2, -4, 6], [1, -1, 1], [1, -3, 4]]),
        #     Matrix([[4, -2, -4, 2], [6, -3, 0, -5], [8, -4, 28, -44], [-8, 4, -4, 12]]),  #
        #     Matrix([[1, -2, 3], [2, 2, 0], [0, -3, 4], [1, 0, 1]]),
        #     Matrix([[3, -3, 3], [-1, -5, 2], [0, -4, 2], [3, -1, 2]]),
        #     Matrix([[1, -1, 1, -3], [2, -1, 4, -2]]),
        #     Matrix([[2, -3, 4, -1], [6, 1, -8, 9], [2, 6, 1, -4]]),
        #     Matrix([[1, 6, 4], [2, 4, -1], [-1, 2, 5]]),
        Matrix(
            [
                [1, 1 / 3, 1 / 2, 1 / 4, 0],
                [0, 0, 0, 1 / 4, 0],
                [0, 0, 0, 1 / 4, 0],
                [0, 1 / 3, 1 / 2, 0, 0],
                [0, 1 / 3, 0, 1 / 4, 1],
            ]
        )
    ]

    b_list: list[list] = [
        #     [-1, 1, -1],
        #     [1, 6, 7, 9],
        #     [6, 17, 12, 7],
        #     [1, 1, 2],
        #     [3, 1, 5],
        #     [8, -1, 0],
        #     [1, 3, 11, -5],
        #     [-3, 0, 1, -1],
        #     [-3, 4, 2, -4],
        #     [0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    # solutions: list[list] = [
    #     [4, -3, -1],
    #     [[9, 4, 0], [-5, -3, 1]],
    #     [2, 3, -2, -1],
    #     [],
    #     [[5.0 / 7, 8.0 / 7, 0], [0, 1, 1]],
    #     [3, 13, 9],
    #     [[1.0 / 2, 0, 1.0 / 4, 0], [1.0 / 2, 1, 0, 0], [5.0 / 6, 0.0, 4.0 / 3, 1]],  #
    #     [-5, 5, 4],
    #     [[-3.0 / 2, -1.0 / 2, 0.0], [-1.0 / 2, 1.0 / 2, 1]],
    #     [[0, 0, 0, 0], [-3, -2, 1, 0], [-1, -4, 0, 1]],
    #     [
    #         [0, 0, 0, 0],
    #         [-19.0 / 50, 16.0 / 25, 23.0 / 25, 1],
    #     ],
    #     [[0, 0, 0], [11.0 / 4, -9.0 / 8, 2]],
    # ]
    n = len(matrices)

    # correct_list = []
    for i in range(n):
        matrices[i] = matrices[i] - get_identity_matrix(matrices[i].rows)
        matrices[i] = matrices[i].addColumn(b_list[i])
        matrix = Gauss_elimination(matrices[i])
        solution = back_substitution(matrix)
        print("Nghiệm của hệ phương trình câu " + str(i + 1) + ":")
        # if roundSol(solution, 2) == roundSol(solutions[i], 2):
        #     correct_list.append(i + 1)
        print(solution)

    # if len(correct_list) == n:
    #     print("Tất cả các ma trận đều đúng")

    ##############################

    # matrices: list[Matrix] = [
    #     Matrix([[1, 2, 1], [3, 7, 3], [2, 3, 4]]),
    #     Matrix([[1, -1, 2], [1, 1, -2], [1, 1, 4]]),
    #     Matrix([[1, 2, 3], [2, 5, 3], [1, 0, 8]]),
    #     Matrix([[-1, 3, -4], [2, 4, 1], [-4, 2, -9]]),
    # ]

    # n = len(matrices)

    # for i in range(n):
    #     print("*** Ma trận câu " + str(i + 1) + ":")
    #     print(matrices[i])

    #     print(">>> Ma trận nghịch đảo tương ứng:")
    #     res = inverse(matrices[i])
    #     if res is not None:
    #         print(res)
    #     else:
    #         print("!!! Ma trận không khả nghịch")

    # a: Matrix = Matrix(
    #     [
    #         [2, 3, -2, 4, -2],
    #         [-1, 2, 3, -1, 1],
    #         [3, -4, -2, 4, 2],
    #         [1, 1, -3, 1, 1],
    #         [0, 0, 5, 0, -2],
    #     ]
    # )
    # a = a.addColumn([1, 8, 7, 3, -1])
    # print(Gauss_elimination(a))
    # print(back_substitution(Gauss_elimination(a)))

    # pi_0: Matrix = Matrix([[26 / 50, 17 / 50, 5 / 50, 2 / 50]])
    # pi_0 = pi_0.transpose()
    # P: Matrix = Matrix(
    #     [
    #         [26 / 34, 0, 0, 0],
    #         [8 / 34, 9 / 12, 0, 0],
    #         [0, 3 / 12, 2 / 3, 0],
    #         [0, 0, 1 / 3, 1],
    #     ]
    # )
    # sigma = 1e-6
    # pi_1 = P * pi_0
    # # print(pow(P, 50) * pi_0)
    # for i in range(100):
    #     pi_i = pow(P, i) * pi_0
    #     p = pi_i[3][0]
    #     if 1 - p < sigma:
    #         print(i)
    #         break


if __name__ == "__main__":
    main()
