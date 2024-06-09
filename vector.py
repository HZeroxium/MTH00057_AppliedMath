class Vector:
    def __init__(self, data: list) -> None:
        self.data = data
        self.dimensions = len(data)

    def __str__(self) -> str:
        for i in range(self.dimensions):
            self.data[i] = round(self.data[i], 10)
        return str(self.data) + "\n"

    def __repr__(self) -> str:
        return f"Vector({self.data})"

    def __add__(self, other: "Vector") -> "Vector":
        if self.dimensions != other.dimensions:
            raise ValueError("Vectors must have the same dimensions")
        return Vector([self.data[i] + other.data[i] for i in range(self.dimensions)])

    def __sub__(self, other: "Vector") -> "Vector":
        if self.dimensions != other.dimensions:
            raise ValueError("Vectors must have the same dimensions")
        return Vector([self.data[i] - other.data[i] for i in range(self.dimensions)])

    def scale(self, scalar: float) -> "Vector":
        return Vector([x * scalar for x in self.data])

    def calDotProduct(self, other) -> float:
        if self.dimensions != other.dimensions:
            raise ValueError("Vectors must have the same dimensions")
        return sum([self.data[i] * other.data[i] for i in range(self.dimensions)])

    def magnitude(self) -> float:
        return sum([x**2 for x in self.data]) ** 0.5

    def unitize(self) -> "Vector":
        return self.scale(1 / self.magnitude())

    def __iter__(self):
        return iter(self.data)


def gram_schmidt(vectors: list[Vector]) -> list[Vector]:
    basis: list[Vector] = []
    for u in vectors:
        temp = Vector([0] * u.dimensions)
        if len(basis) == 0:
            basis.append(u)
            continue
        for v in basis:
            temp = temp + v.scale(u.calDotProduct(v) / v.calDotProduct(v))
        temp = u - temp
        if temp.magnitude() != 0:
            basis.append(temp)
        else:
            raise ValueError("Vectors must be linearly independent")
    return basis
