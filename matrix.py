# Define a class Matrix that represents a matrix.
# The matrix should have the following attributes: rows, columns and data.


class Matrix:
    def __init__(self, data: list[list]) -> None:
        self.data = data
        self.rows = len(data)
        self.columns = len(data[0])

    def __str__(self) -> str:
        result = ""
        for i in range(self.rows):
            for j in range(self.columns):
                self.data[i][j] = round(self.data[i][j], 10)
            result += str(self.data[i]) + "\n"
        return result

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def __getitem__(self, index: int) -> list:
        return self.data[index]

    def __setitem__(self, index: int, value: list):
        self.data[index] = value

    def __add__(self, other) -> "Matrix":
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Matrices must have the same dimensions")
        new_data = []
        for i in range(self.rows):
            new_data.append(
                [self.data[i][j] + other.data[i][j] for j in range(self.columns)]
            )
        return Matrix(new_data)

    def __sub__(self, other) -> "Matrix":
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Matrices must have the same dimensions")
        new_data = []
        for i in range(self.rows):
            new_data.append(
                [self.data[i][j] - other.data[i][j] for j in range(self.columns)]
            )
        return Matrix(new_data)

    def __mul__(self, other: "Matrix") -> "Matrix":
        new_data = []
        if type(other) == Matrix:
            if self.columns != other.rows:
                raise ValueError(
                    "The number of columns in the first matrix must be equal to the number of rows in the second matrix"
                )
            for i in range(self.rows):
                new_data.append(
                    [
                        sum(
                            [
                                self.data[i][k] * other.data[k][j]
                                for k in range(self.columns)
                            ]
                        )
                        for j in range(other.columns)
                    ]
                )
        elif type(other) == int or type(other) == float:
            new_data = [[x * other for x in row] for row in self.data]
        return Matrix(new_data)

    def getColumn(self: "Matrix", index: int) -> list:
        return [self.data[i][index] for i in range(self.rows)]

    def getRow(self: "Matrix", index: int) -> list:
        return self.data[index]

    def transpose(self) -> "Matrix":
        new_data = []
        for j in range(self.columns):
            new_data.append([self.data[i][j] for i in range(self.rows)])
        return Matrix(new_data)

    def getMinor(self: "Matrix", i: int, j: int) -> "Matrix":
        new_data = [
            [self.data[x][y] for y in range(self.columns) if y != j]
            for x in range(self.rows)
            if x != i
        ]
        return Matrix(new_data)

    def getCofactor(self: "Matrix", i: int, j: int) -> float:
        return self.getMinor(i, j).getDeterminant() * (-1) ** (i + j)

    def getDeterminant(self: "Matrix") -> float:
        if self.rows != self.columns:
            raise ValueError("The matrix must be square")
        if self.rows == 1:
            return self.data[0][0]
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        determinant = 0
        for i in range(self.rows):
            determinant += (
                self.data[0][i] * self.getMinor(0, i).getDeterminant() * (-1) ** i
            )
        return determinant

    def getInverse(self: "Matrix") -> "Matrix":
        determinant = self.getDeterminant()
        if determinant == 0:
            raise ValueError("The matrix is not invertible")
        new_data = [
            [self.getCofactor(i, j) / determinant for j in range(self.columns)]
            for i in range(self.rows)
        ]
        return Matrix(new_data)

    def addColumn(self: "Matrix", column: list) -> "Matrix":
        if len(column) != self.rows:
            raise ValueError("The column must have the same length as the matrix")
        new_data = [self.data[i] + [column[i]] for i in range(self.rows)]
        return Matrix(new_data)

    def separateColumn(self: "Matrix", index: int) -> tuple["Matrix", list]:
        column = self.getColumn(index)
        new_data = [
            self.data[i][:index] + self.data[i][index + 1 :] for i in range(self.rows)
        ]
        return Matrix(new_data), column

    def addRow(self: "Matrix", row: list) -> "Matrix":
        if len(row) != self.columns:
            raise ValueError("The row must have the same length as the matrix")
        new_data = self.data + [row]
        return Matrix(new_data)

    def removeRow(self: "Matrix", index: int) -> "Matrix":
        new_data = self.data[:index] + self.data[index + 1 :]
        return Matrix(new_data)

    def swapRows(self: "Matrix", i: int, j: int) -> None:
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def swapColumns(self: "Matrix", i: int, j: int) -> None:
        for k in range(self.rows):
            self.data[k][i], self.data[k][j] = self.data[k][j], self.data[k][i]

    def attach_matrix_horizontal(self: "Matrix", other: "Matrix") -> "Matrix":
        if self.rows != other.rows:
            raise ValueError("Matrices must have the same number of rows")
        new_data = [self.data[i] + other.data[i] for i in range(self.rows)]
        return Matrix(new_data)

    def separate_matrix_horizontal(
        self: "Matrix", index: int
    ) -> tuple["Matrix", "Matrix"]:
        new_data1 = [self.data[i][:index] for i in range(self.rows)]
        new_data2 = [self.data[i][index:] for i in range(self.rows)]
        return Matrix(new_data1), Matrix(new_data2)

    def is_identity(self: "Matrix") -> bool:
        if self.rows != self.columns:
            return False
        for i in range(self.rows):
            for j in range(self.columns):
                if i == j:
                    if self.data[i][j] != 1.0:
                        return False
                else:
                    if self.data[i][j] != 0.0:
                        return False
        return True

    def Guass_Jordan(self: "Matrix") -> "Matrix":
        for i in range(self.rows):
            if self.data[i][i] == 0:
                for j in range(i + 1, self.rows):
                    if self.data[j][i] != 0:
                        self.swapRows(i, j)
                        break
            if self.data[i][i] == 0:
                continue
            self.data[i] = [x / self.data[i][i] for x in self.data[i]]
            for j in range(self.rows):
                if j != i:
                    self.data[j] = [
                        self.data[j][k] - self.data[j][i] * self.data[i][k]
                        for k in range(self.columns)
                    ]
        return self

    def getRank(self: "Matrix") -> int:
        rank = 0
        for i in range(self.rows):
            if all([x == 0 for x in self.data[i]]):
                break
            rank += 1
        return rank

    def getNullity(self: "Matrix") -> int:
        return self.columns - self.getRank()

    def format(self: "Matrix") -> "Matrix":
        for i in range(self.rows):
            for j in range(self.columns):
                if type(self.data[i][j]) == int:
                    self.data[i][j] = float(self.data[i][j])
                if abs(self.data[i][j]) < 1e-10:
                    self.data[i][j] = 0
                if self.data[i][j] == -0.0:
                    self.data[i][j] = 0.0
                self.data[i][j] = round(self.data[i][j], 2)
        return self

    def trace(self: "Matrix") -> float:
        if self.rows != self.columns:
            raise ValueError("The matrix must be square")
        return sum([self.data[i][i] for i in range(self.rows)])

    def is_square(self: "Matrix") -> bool:
        return self.rows == self.columns


def Gauss_elimination(matrix: "Matrix") -> "Matrix":
    """
    Performs Gaussian elimination on the given matrix.

    Args:
        matrix (Matrix): The matrix to perform Gaussian elimination on.

    Returns:
        Matrix: The matrix after Gaussian elimination has been applied.
    """
    for i in range(matrix.rows):
        pivot = matrix.data[i][i]
        if pivot == 0:
            for j in range(i + 1, matrix.rows):
                if matrix.data[j][i] != 0:
                    matrix.swapRows(i, j)
                    pivot = matrix.data[i][i]
                    break
        if pivot == 0:
            continue
        matrix.data[i] = [x / pivot for x in matrix.data[i]]
        for j in range(i + 1, matrix.rows):
            matrix.data[j] = [
                matrix.data[j][k] - matrix.data[j][i] * matrix.data[i][k]
                for k in range(matrix.columns)
            ]
    return matrix


def back_substitution(matrix: Matrix) -> list:
    """
    Perform back substitution to solve a system of linear equations represented by a matrix.

    Args:
        matrix (Matrix): The matrix representing the system of linear equations.

    Returns:
        list: The solution(s) to the system of linear equations. If the system has a unique solution,
            a list containing the values of the variables is returned. If the system has infinitely
            many solutions, a list of solution vectors is returned.
    Raises:
        None
    """
    matrix_a, _ = matrix.separateColumn(matrix.columns - 1)
    rankA = matrix_a.getRank()
    rankA_ = matrix.getRank()

    if rankA < rankA_:
        # print(">>> Hệ phương trình vô nghiệm")
        return []

    n = matrix.columns - 1
    free_variables = []

    while matrix.rows < matrix.columns - 1:
        matrix = matrix.addRow([0] * matrix.columns)

    while matrix.rows > matrix.columns - 1:
        matrix = matrix.removeRow(matrix.rows - 1)

    # Currently, the matrix is ​​in the form of the above triangular matrix, let's convert it into a main diagonal matrix
    for i in range(matrix.rows):
        if matrix.data[i][i] == 1:
            for j in range(i):
                matrix.data[j] = [
                    matrix.data[j][k] - matrix.data[j][i] * matrix.data[i][k]
                    for k in range(matrix.columns)
                ]
        else:
            if i <= n:
                free_variables.append(i)

    rankA_ = matrix.getRank()
    # Unique solution
    if rankA == rankA_ and rankA == n:
        # print(">>> Hệ phương trình có nghiệm duy nhất")
        return [matrix[i][n] for i in range(n)]

    # Infinite solutions
    sol = []
    temp = [0] * n
    for i in range(n):
        if i < matrix.rows:
            temp[i] = matrix.data[i][n]
    sol.append(temp)

    # Consider columns that contain free hidden content
    for i in free_variables:
        temp = [0] * n
        temp[i] = 1
        for j in range(n):
            if j != i:
                temp[j] = -matrix.data[j][i]
        sol.append(temp)

    # print(">>> Hệ phương trình có vô số nghiệm")
    return sol


def get_identity_matrix(n: int) -> Matrix:
    data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    return Matrix(data)


def inverse(A: Matrix) -> Matrix:
    """
    Find the inverse of a square matrix using the Gauss-Jordan method.

    Args:
        A (Matrix): The matrix to find the inverse of.

    Returns:
        Matrix: The inverse of the input matrix A.
    """
    if A.rows != A.columns:
        raise ValueError("The input matrix must be square")

    n = A.rows
    I = get_identity_matrix(n)
    augmented_matrix = A.attach_matrix_horizontal(I)
    reduced_matrix = augmented_matrix.Guass_Jordan()
    _, inverse_matrix = reduced_matrix.separate_matrix_horizontal(n)

    if not _.is_identity():
        return None
    return inverse_matrix
