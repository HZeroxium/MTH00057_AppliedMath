import itertools
import pandas as pd
from pandas import DataFrame, Series
from matrix import Matrix
from typing import List
import random
import math
from model import (
    LinearRegression,
    PolynomialRegression,
    InteractionTermsRegression,
    RidgeRegression,
)
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from tabulate import tabulate

models = {
    "Linear Regression": LinearRegression(),
    # "Polynomial Regression (degree=2)": PolynomialRegression(degree=2),
    # "Polynomial Regression (degree=3)": PolynomialRegression(degree=3),
    "Interaction Terms Regression": InteractionTermsRegression(),
    "Ridge Regression": RidgeRegression(alpha=1.0),
}

split_ratios = [0.4, 0.6, 0.8, 0.9]
num_runs = 10
shuffle = True
plot_result = False


def load_data():
    """Load and preprocess the data."""
    data = pd.read_csv("NHANES_age_prediction.csv", delimiter=",")
    data = data.drop(columns=["SEQN", "age_group"]).dropna()

    features = [
        "RIDAGEYR",
        "RIAGENDR",
        "PAQ605",
        "BMXBMI",
        "LBXGLU",
        "LBXGLT",
        "LBXIN",
    ]
    target = "DIQ010"

    X: DataFrame = data[features]
    y: Series = data[target]

    return X, y, features


def train_test_split(X, y, test_size=0.2, shuffle=True):
    """Split the data into training and testing sets."""
    indices = list(range(X.rows))

    if shuffle:
        random.shuffle(indices)

    test_indices = indices[: math.floor(test_size * len(indices))]
    train_indices = indices[math.floor(test_size * len(indices)) :]

    X_train = Matrix([X[i] for i in train_indices])
    y_train = Matrix([y[i] for i in train_indices])
    X_test = Matrix([X[i] for i in test_indices])
    y_test = Matrix([y[i] for i in test_indices])

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train the model and evaluate using Mean Squared Error (MSE)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = model.mean_squared_error(y_test, y_pred)
    return mse


def format_results(results):
    """Format the results into a readable table format."""
    formatted_results = pd.DataFrame(results).T
    formatted_results.columns = [
        "Split 60/40",
        "Split 40/60",
        "Split 20/80",
        "Split 10/90",
    ]
    return formatted_results


def plot_results(results, title):
    """Visualize the results using seaborn."""
    sns.set(style="whitegrid")
    df_results = format_results(results).reset_index()
    df_results = pd.melt(
        df_results, id_vars="index", var_name="Split Ratio", value_name="MSE"
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Split Ratio", y="MSE", hue="index", data=df_results)
    plt.title(title)
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Train/Test Split Ratio")
    plt.legend(title="Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def run_models_with_different_splits(
    X: DataFrame,
    y: Series,
    split_ratios=[0.4, 0.6, 0.8, 0.9],
    num_runs=10,
    models={
        "Linear Regression": LinearRegression(),
    },
    plot_result=True,
    shuffle=True,
):
    """Run models with different train/test splits and visualize the results."""
    if not shuffle:
        num_runs = 1
    X_list = X.to_numpy().tolist()
    y_list = y.to_numpy().tolist()
    X_matrix = Matrix(X_list)
    y_matrix = Matrix([y_list]).transpose()
    results = {model_name: [] for model_name in models}

    for split_ratio in split_ratios:
        table_data = []
        headers = ["Model", "Split Ratio", "MSE"]

        for model_name, model in models.items():
            mse_list = []
            for _ in range(num_runs):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_matrix, y_matrix, test_size=split_ratio, shuffle=shuffle
                )
                mse = evaluate_model(model, X_train, y_train, X_test, y_test)
                mse_list.append(mse)

            avg_mse = sum(mse_list) / num_runs
            results[model_name].append(avg_mse)
            table_data.append(
                [
                    model_name,
                    f"{int((1 - split_ratio) * 100)}/{int(split_ratio * 100)}",
                    f"{avg_mse:.4f}",
                ]
            )

        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    formatted_results = format_results(results)

    # Plot the results for better visualization
    if plot_result:
        plot_results(
            results,
            "Mean Squared Error (MSE) across Different Models and Train/Test Splits",
        )

    # Tìm mô hình tốt nhất dựa trên MSE trung bình
    avg_results = {
        model_name: sum(mse_list) / len(mse_list)
        for model_name, mse_list in results.items()
    }
    best_model_name = min(avg_results, key=avg_results.get)

    summary_table = [[best_model_name, f"{avg_results[best_model_name]:.4f}"]]
    print(tabulate(summary_table, headers=["Best Model", "MSE"], tablefmt="pretty"))


def run_models_with_single_feature(
    X: DataFrame,
    y: Series,
    features,
    split_ratios=[0.4, 0.6, 0.8, 0.9],
    num_runs=10,
    models={
        "Linear Regression": LinearRegression(),
    },
    plot_result=True,
    shuffle=True,
):
    """Run models with different train/test splits for each single feature."""
    if not shuffle:
        num_runs = 1
    models.pop("Interaction Terms Regression", None)
    y_list = y.to_numpy().tolist()
    y_matrix = Matrix([y_list]).transpose()

    final_results = {}

    for feature_idx, feature in enumerate(features):
        table_data = []
        headers = ["Model", "Feature", "Split Ratio", "MSE"]

        X_single_feature = X[feature]  # Chỉ lấy một cột
        X_list = X_single_feature.to_numpy().tolist()
        X_matrix = Matrix([X_list]).transpose()
        results = {model_name: [] for model_name in models}

        for split_ratio in split_ratios:
            for model_name, model in models.items():
                mse_list = []
                for _ in range(num_runs):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_matrix, y_matrix, test_size=split_ratio, shuffle=shuffle
                    )
                    mse = evaluate_model(model, X_train, y_train, X_test, y_test)
                    mse_list.append(mse)

                avg_mse = sum(mse_list) / num_runs
                results[model_name].append(avg_mse)
                table_data.append(
                    [
                        model_name,
                        feature,
                        f"{int((1 - split_ratio) * 100)}/{int(split_ratio * 100)}",
                        f"{avg_mse:.4f}",
                    ]
                )

        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

        final_results[feature] = results

    # Plot results for each feature
    if plot_result:
        for feature, result in final_results.items():
            formatted_results = format_results(result)
            plot_results(result, f"Mean Squared Error (MSE) for {feature}")

    # Tìm ra mô hình tốt nhất cho mỗi feature
    best_results = {}
    for feature, result in final_results.items():
        avg_results = {
            model_name: sum(mse_list) / len(mse_list)
            for model_name, mse_list in result.items()
        }
        best_model_name = min(avg_results, key=avg_results.get)
        best_results[feature] = (best_model_name, avg_results[best_model_name])

    summary_table = []
    for feature, (model_name, mse) in best_results.items():
        summary_table.append([feature, model_name, f"{mse:.4f}"])

    print(
        tabulate(
            summary_table, headers=["Feature", "Best Model", "MSE"], tablefmt="pretty"
        )
    )


def run_models_with_feature_combinations(
    X: DataFrame,
    y: Series,
    features,
    split_ratios=[0.4, 0.6, 0.8, 0.9],
    num_runs=10,
    models={
        "Linear Regression": LinearRegression(),
    },
    plot_result=True,
    shuffle=True,
):
    """Run models with different train/test splits for all feature combinations."""
    if not shuffle:
        num_runs = 1
    y_list = y.to_numpy().tolist()
    y_matrix = Matrix([y_list]).transpose()

    final_results = {}

    for r in range(
        1, len(features) + 1
    ):  # Tạo ra tất cả các tổ hợp từ 1 đến 7 đặc trưng
        for combination in itertools.combinations(features, r):
            table_data = []
            headers = ["Model", "Feature Combination", "Split Ratio", "MSE"]

            print(f"\nEvaluating combination: {combination}")
            X_combination = X[list(combination)]  # Chỉ lấy các cột trong tổ hợp
            X_list = X_combination.to_numpy().tolist()
            X_matrix = Matrix(X_list)
            results = {model_name: [] for model_name in models}

            for split_ratio in split_ratios:
                for model_name, model in models.items():
                    mse_list = []
                    for _ in range(num_runs):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_matrix, y_matrix, test_size=split_ratio, shuffle=shuffle
                        )
                        mse = evaluate_model(model, X_train, y_train, X_test, y_test)
                        mse_list.append(mse)

                    avg_mse = sum(mse_list) / num_runs
                    results[model_name].append(avg_mse)
                    table_data.append(
                        [
                            model_name,
                            str(combination),
                            f"{int((1 - split_ratio) * 100)}/{int(split_ratio * 100)}",
                            f"{avg_mse:.4f}",
                        ]
                    )

            print(tabulate(table_data, headers=headers, tablefmt="pretty"))

            final_results[combination] = results

    # Plot results for each combination
    if plot_result:
        for combination, result in final_results.items():
            formatted_results = format_results(result)
            plot_results(result, f"Mean Squared Error (MSE) for {combination}")

    # Tìm ra mô hình tốt nhất cho mỗi combination
    best_results = {}
    for combination, result in final_results.items():
        avg_results = {
            model_name: sum(mse_list) / len(mse_list)
            for model_name, mse_list in result.items()
        }
        best_model_name = min(avg_results, key=avg_results.get)
        best_results[combination] = (best_model_name, avg_results[best_model_name])

    # In ra kết quả tốt nhất cho từng combination
    summary_table = []
    for combination, (model_name, mse) in best_results.items():
        summary_table.append([str(combination), model_name, f"{mse:.4f}"])

    print(
        tabulate(
            summary_table,
            headers=["Combination", "Best Model", "MSE"],
            tablefmt="pretty",
        )
    )

    # Tìm ra tổ hợp tốt nhất
    best_combination = min(best_results, key=lambda x: best_results[x][1])
    best_model, best_mse = best_results[best_combination]
    print(
        f"\nBest combination: {best_combination} | Best Model: {best_model} | MSE: {best_mse:.4f}"
    )


def main():
    """Main function to load data, run models, and visualize results."""
    X, y, features = load_data()

    # Chạy mô hình và so sánh kết quả với các tỷ lệ chia khác nhau cho tất cả các đặc trưng
    run_models_with_different_splits(
        X,
        y,
        models=models,
        num_runs=num_runs,
        models=models,
        shuffle=shuffle,
        plot_result=plot_result,
    )

    # Chạy mô hình và so sánh kết quả với từng đặc trưng
    run_models_with_single_feature(
        X,
        y,
        features=features,
        models=models,
        num_runs=num_runs,
        shuffle=shuffle,
        plot_result=plot_result,
    )

    # Chạy mô hình và so sánh kết quả với các tổ hợp đặc trưng
    run_models_with_feature_combinations(
        X,
        y,
        features=features,
        models=models,
        num_runs=num_runs,
        shuffle=shuffle,
        plot_result=plot_result,
    )


if __name__ == "__main__":
    main()
