from matrix import Matrix
import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu đầu vào
data = Matrix([[34, 26], [12, 17], [3, 5], [1, 2]])


def calculate_transition_matrix_and_initial_distribution(
    data: Matrix,
) -> tuple[Matrix, Matrix, int]:
    """
    Tính toán ma trận chuyển trạng thái và vector phân phối xác suất ban đầu.

    Args:
        data (Matrix): Ma trận dữ liệu đầu vào, mỗi hàng đại diện cho một trạng thái với
                       số lượng phòng học ở hai thời điểm khác nhau.

    Returns:
        tuple[Matrix, Matrix, int]: Trả về ma trận chuyển trạng thái P, vector phân phối xác suất ban đầu pi_0,
                                    và tổng số phòng học.
    """
    num_states = data.rows

    # Khởi tạo ma trận P với các giá trị bằng 0
    P_data = [[0] * num_states for _ in range(num_states)]

    # Tính toán xác suất chuyển trạng thái cho mỗi trạng thái
    for i in range(num_states):
        if i == 0:
            P_data[i][i] = data[i][1] / data[i][0]  # Giữ nguyên trạng thái đầu tiên
            if num_states > 1:
                P_data[i + 1][i] = (data[i][0] - data[i][1]) / data[i][
                    0
                ]  # Chuyển từ trạng thái đầu tiên sang trạng thái tiếp theo
        elif i < num_states - 1:
            rooms_down = sum([data[j][0] - data[j][1] for j in range(i)]) + (
                data[i][0] - data[i][1]
            )
            P_data[i][i] = (data[i][0] - rooms_down) / data[i][
                0
            ]  # Giữ nguyên trạng thái hiện tại
            P_data[i + 1][i] = (
                rooms_down / data[i][0]
            )  # Chuyển xuống trạng thái tiếp theo
        else:
            P_data[i][i] = 1  # Trạng thái cuối cùng, chỉ có thể giữ nguyên

    P = Matrix(P_data)

    # Tính vector phân phối xác suất ban đầu pi_0
    total_rooms = sum([data[i][1] for i in range(num_states)])
    pi_0_data = [[data[i][1] / total_rooms] for i in range(num_states)]
    pi_0 = Matrix(pi_0_data)

    return P, pi_0, total_rooms


# Tính toán ma trận chuyển trạng thái P và vector phân phối xác suất ban đầu pi_0
P, pi_0, total_rooms = calculate_transition_matrix_and_initial_distribution(data)

# In kết quả
print("Ma trận chuyển trạng thái P:")
print(P)
print("\nVector phân phối xác suất ban đầu pi_0:")
print(pi_0)


def forecast_degradation(P: Matrix, pi_0: Matrix, years: int = 15) -> pd.DataFrame:
    """
    Dự báo sự xuống cấp của CSVC trong vòng 'years' năm tới.

    Args:
        P (Matrix): Ma trận chuyển trạng thái.
        pi_0 (Matrix): Vector phân phối xác suất ban đầu.
        years (int): Số năm để dự báo (mặc định là 15).

    Returns:
        pd.DataFrame: Bảng chứa phân phối xác suất cho từng năm.
    """
    forecast = []
    pi_t = pi_0
    for t in range(years + 1):  # Bao gồm năm hiện tại
        forecast.append([round(x[0], 4) for x in pi_t.data])
        pi_t = P * pi_t

    # Tạo DataFrame để hiển thị kết quả
    years_list = [2021 + i for i in range(years + 1)]
    df_forecast = pd.DataFrame(
        forecast, columns=[f"Trạng thái {i+1}" for i in range(P.rows)], index=years_list
    )

    return df_forecast


def plot_forecast(df_forecast: pd.DataFrame) -> None:
    """
    Vẽ biểu đồ dự báo sự xuống cấp của CSVC.

    Args:
        df_forecast (pd.DataFrame): DataFrame chứa kết quả dự báo.
    """
    df_forecast.plot(kind="line", figsize=(10, 6), marker="o")
    plt.title("Dự báo sự xuống cấp của CSVC các phòng học tòa nhà I")
    plt.xlabel("Năm")
    plt.ylabel("Xác suất trạng thái")
    plt.grid(True)
    plt.show()


# Sử dụng hàm để dự báo và vẽ biểu đồ
df_forecast = forecast_degradation(P, pi_0, years=15)
print(df_forecast)
plot_forecast(df_forecast)


def find_rebuild_year(P: Matrix, pi_0: Matrix, total_rooms: int) -> int:
    """
    Xác định thời điểm cần phá hủy/trùng tu tòa nhà khi tất cả phòng học đều ở trạng thái xấu.

    Args:
        P (Matrix): Ma trận chuyển trạng thái.
        pi_0 (Matrix): Vector phân phối xác suất ban đầu.
        total_rooms (int): Tổng số phòng học.

    Returns:
        int: Năm cần phá hủy/trùng tu.
    """
    pi_t = pi_0
    year = 2021

    while True:
        if (
            round(pi_t[-1][0] * total_rooms, ndigits=4) == total_rooms
        ):  # Tất cả phòng học đều ở trạng thái 4
            return year
        pi_t = P * pi_t
        year += 1


# Sử dụng hàm để xác định năm cần phá hủy/trùng tu
rebuild_year = find_rebuild_year(P, pi_0, total_rooms)
print(f"Năm cần phá hủy/trùng tu tòa nhà: {rebuild_year}")


def calculate_lifetime(rebuild_year: int, start_year: int = 2021) -> int:
    """
    Xác định tuổi thọ của tòa nhà I.

    Args:
        rebuild_year (int): Năm phá hủy/trùng tu.
        start_year (int): Năm bắt đầu theo dõi (mặc định là 2021).

    Returns:
        int: Tuổi thọ của tòa nhà I.
    """
    return rebuild_year - start_year


# Sử dụng hàm để xác định tuổi thọ của tòa nhà
lifetime = calculate_lifetime(rebuild_year)
print(f"Tuổi thọ của tòa nhà I: {lifetime} năm")
