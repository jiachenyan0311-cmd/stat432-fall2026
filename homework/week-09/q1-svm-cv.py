import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURES = [
    "sbp",
    "tobacco",
    "ldl",
    "adiposity",
    "famhist",
    "typea",
    "obesity",
    "alcohol",
    "age",
]


def candidates():
    result = []
    for cost in [0.01, 0.1, 1, 10]:
        result.append(
            {
                "kernel": "linear",
                "degree": None,
                "C": cost,
                "gamma": None,
                "coef0": None,
            }
        )
    for degree in [2, 3, 4]:
        for cost in [0.1, 1, 10]:
            result.append(
                {
                    "kernel": "polynomial",
                    "degree": degree,
                    "C": cost,
                    "gamma": 1 / 9,
                    "coef0": 1,
                }
            )
    for cost, gamma in [
        (0.1, 0.01),
        (1, 0.01),
        (100, 0.01),
        (0.1, 0.1),
        (1, 0.1),
        (10, 0.1),
        (1, 1),
    ]:
        result.append(
            {
                "kernel": "RBF",
                "degree": None,
                "C": cost,
                "gamma": gamma,
                "coef0": None,
            }
        )
    return result


def make_estimator(setting):
    kernel = {"polynomial": "poly", "RBF": "rbf"}.get(
        setting["kernel"], setting["kernel"]
    )
    parameters = {
        "kernel": kernel,
        "C": setting["C"],
    }
    if setting["gamma"] is not None:
        parameters["gamma"] = setting["gamma"]
    if setting["degree"] is not None:
        parameters["degree"] = setting["degree"]
        parameters["coef0"] = setting["coef0"]
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC(**parameters)),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_csv", type=Path)
    args = parser.parse_args()
    script_directory = Path(__file__).resolve().parent
    output_directory = script_directory / "results"
    output_directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data_csv)
    if len(data) != 462:
        raise ValueError(f"Expected 462 observations, found {len(data)}.")
    if data[FEATURES + ["chd"]].isna().any().any():
        raise ValueError("Missing values found; Q1 does not specify imputation.")
    if set(data["famhist"].unique()) != {"Absent", "Present"}:
        raise ValueError("Unexpected famhist levels.")
    if set(data["chd"].unique()) != {0, 1}:
        raise ValueError("Expected chd coded as 0/1.")

    X = data[FEATURES].copy()
    X["famhist"] = X["famhist"].map({"Absent": 1, "Present": 2})
    y = data["chd"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=432)
    folds = list(cv.split(X, y))
    rows = []
    for index, setting in enumerate(candidates(), start=1):
        fold_errors = []
        for train_index, validation_index in folds:
            model = make_estimator(setting)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            prediction = model.predict(X.iloc[validation_index])
            fold_errors.append(
                (prediction != y.iloc[validation_index].to_numpy()).mean()
            )
        row = {
            "candidate": index,
            **setting,
            "fold_1_error": fold_errors[0],
            "fold_2_error": fold_errors[1],
            "fold_3_error": fold_errors[2],
            "fold_4_error": fold_errors[3],
            "fold_5_error": fold_errors[4],
            "mean_cv_error": sum(fold_errors) / len(fold_errors),
        }
        rows.append(row)
        print(
            index,
            setting["kernel"],
            setting["degree"],
            setting["C"],
            setting["gamma"],
            f"{row['mean_cv_error']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_directory / "q1-cv-results.csv", index=False)
    (output_directory / "q1-folds.csv").write_text(
        pd.DataFrame(
            [
                {
                    "observation_row": int(row_index) + 1,
                    "fold": fold_number,
                }
                for fold_number, (_, validation_index) in enumerate(folds, start=1)
                for row_index in validation_index
            ]
        ).sort_values("observation_row").to_csv(index=False),
        encoding="utf-8",
    )
    best = results.loc[results["mean_cv_error"].idxmin()]
    print("Candidates:", len(results))
    print("Best candidate:", best.to_dict())


if __name__ == "__main__":
    main()
