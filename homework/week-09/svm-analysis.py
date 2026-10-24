import argparse
import json
import platform
import sys
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import reportlab
import sklearn
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
FAMILY_ORDER = ["linear", "polynomial", "rbf"]
FAMILY_PRIORITY = {"linear": 0, "polynomial": 1, "rbf": 2}
C_VALUES = [0.1, 1.0, 10.0]


def candidate_grids():
    common = Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC()),
        ]
    )
    return {
        "linear": (
            common,
            {
                "svc__kernel": ["linear"],
                "svc__C": C_VALUES,
            },
        ),
        "polynomial": (
            common,
            {
                "svc__kernel": ["poly"],
                "svc__C": C_VALUES,
                "svc__degree": [2, 3, 4],
                "svc__gamma": [0.1, 1.0],
                "svc__coef0": [1.0],
            },
        ),
        "rbf": (
            common,
            {
                "svc__kernel": ["rbf"],
                "svc__C": C_VALUES,
                "svc__gamma": [0.01, 0.1, 1.0],
            },
        ),
    }


def fit_search(family, X, y, seed):
    estimator, grid = candidate_grids()[family]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    return GridSearchCV(
        estimator,
        grid,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    ).fit(X, y)


def select_with_one_se(searches):
    best_family = max(FAMILY_ORDER, key=lambda name: searches[name].best_score_)
    best_search = searches[best_family]
    best_index = best_search.best_index_
    standard_error = (
        best_search.cv_results_["std_test_score"][best_index] / np.sqrt(10)
    )
    cutoff = best_search.best_score_ - standard_error
    eligible = [
        family
        for family in FAMILY_ORDER
        if searches[family].best_score_ >= cutoff
    ]
    selected = min(eligible, key=FAMILY_PRIORITY.__getitem__)
    return selected, best_family, standard_error


def support_count(estimator):
    return int(np.sum(estimator.named_steps["svc"].n_support_))


def write_roc_grid(oof_scores, y, selected_scores, path):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    panels = [
        ("Linear", oof_scores["linear"]),
        ("Polynomial", oof_scores["polynomial"]),
        ("RBF", oof_scores["rbf"]),
        ("Selected by inner CV", selected_scores),
    ]
    for axis, (label, scores) in zip(axes.flat, panels):
        false_positive, true_positive, _ = roc_curve(y, scores)
        pooled_auc = roc_auc_score(y, scores)
        axis.plot(false_positive, true_positive, color="#2457A7", linewidth=2)
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        axis.set(
            title=f"{label} (pooled OOF AUC={pooled_auc:.3f})",
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Nested 10-fold cross-validation ROC curves", fontsize=11)
    for axis in axes.flat:
        axis.title.set_fontsize(11)
        axis.xaxis.label.set_size(11)
        axis.yaxis.label.set_size(11)
        axis.tick_params(axis="both", labelsize=11)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def draw_paragraph(pdf, text, x, top, width, style):
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, top)
    paragraph.drawOn(pdf, x, top - height)
    return top - height


def draw_report(
    output_pdf,
    report_values,
    plot_path,
    skill_text,
    csv_path,
    environment_path,
):
    appendix_font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    pdfmetrics.registerFont(
        TTFont("SkillAppendix", str(appendix_font_path), subfontIndex=0)
    )
    page_width, page_height = letter
    margin = 34
    content_width = page_width - 2 * margin
    body = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=11,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#202124"),
        spaceAfter=0,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=11,
        leading=13,
    )
    pdf = canvas.Canvas(str(output_pdf), pagesize=letter)
    pdf.setTitle("Week 09 SVM Analysis Report")
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin, page_height - 38, "SVM Analysis: South African Heart Disease")
    y = page_height - 56
    y = draw_paragraph(
        pdf,
        "Response: CHD (0/1); predictors: all 9 listed variables; family history coded "
        "Absent=1 and Present=2. The dataset has 462 observations (302 CHD=0, 160 CHD=1), "
        "with no missing values.",
        margin,
        y,
        content_width,
        body,
    ) - 5
    y = draw_paragraph(
        pdf,
        "Methods: seed 2027; stratified nested 10-fold CV with shared outer folds. "
        "Standardization was fitted inside each training fold. Inner 10-fold CV tuned "
        "C over 0.01-100 for all kernels; RBF gamma over 0.001-1; polynomial degree "
        "over 2-4 and gamma over 0.01-1 (coef0=1). Selection used ROC-AUC and the "
        "one-standard-error rule, preferring linear, then polynomial, then RBF when "
        "within one SE of the best inner-CV score. Balanced accuracy is also reported.",
        margin,
        y,
        content_width,
        body,
    ) - 7

    table_data = [
        [
            "Kernel",
            "Outer ROC-AUC\nmean +/- SD",
            "Balanced acc.\nmean +/- SD",
            "Outer-fold SV\nmean +/- SD",
        ]
    ]
    for family in FAMILY_ORDER:
        summary = report_values["summary"].loc[family]
        table_data.append(
            [
                family.title(),
                f"{summary['roc_auc_mean']:.3f} +/- {summary['roc_auc_sd']:.3f}",
                f"{summary['balanced_accuracy_mean']:.3f} +/- "
                f"{summary['balanced_accuracy_sd']:.3f}",
                f"{summary['support_vectors_mean']:.1f} +/- "
                f"{summary['support_vectors_sd']:.1f}",
            ]
        )
    table_data.append(
        [
            "Selected procedure",
            f"{report_values['selected_auc_mean']:.3f} +/- "
            f"{report_values['selected_auc_sd']:.3f}",
            f"{report_values['selected_ba_mean']:.3f} +/- "
            f"{report_values['selected_ba_sd']:.3f}",
            "See family rows",
        ]
    )
    table = Table(
        table_data,
        colWidths=[120, 120, 140, 130],
        rowHeights=[31, 23, 23, 23, 23],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EEF8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB4C0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    table_width, table_height = table.wrapOn(pdf, content_width, y)
    table.drawOn(pdf, margin, y - table_height)
    y -= table_height + 4

    plot_height = min(214, max(180, y - 260))
    pdf.drawImage(
        str(plot_path),
        margin,
        y - plot_height,
        width=content_width,
        height=plot_height,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )
    y -= plot_height + 3

    selected = report_values["selected_family"]
    params = report_values["selected_params"]
    param_text = ", ".join(
        f"{key.replace('svc__', '')}={value}" for key, value in sorted(params.items())
    )
    selected_text = (
        f"Final model: {selected.title()} SVM ({param_text}); "
        f"{report_values['full_sv']} support vectors "
        f"({report_values['full_sv_ratio']:.1%} of N). "
        f"The outer-CV-selected procedure achieved ROC-AUC "
        f"{report_values['selected_auc_mean']:.3f} +/- "
        f"{report_values['selected_auc_sd']:.3f} and balanced accuracy "
        f"{report_values['selected_ba_mean']:.3f} +/- "
        f"{report_values['selected_ba_sd']:.3f}. Top held-out permutation-importance "
        f"features (mean ROC-AUC decrease): {report_values['top_features']}."
    )
    y = draw_paragraph(
        pdf, selected_text, margin, y, content_width, body
    ) - 4
    y = draw_paragraph(
        pdf,
        "Interpretation: the selected kernel is preferred under the stated nested-CV "
        "selection rule, not proven universally best. Fold-to-fold SD describes split "
        "variation, not a confidence interval. Permutation importance is predictive, "
        "not causal; correlated predictors can share or mask importance. No imputation "
        "was needed. The full-data refit is for deployment/submission, not an independent "
        "test-set estimate.",
        margin,
        y,
        content_width,
        body,
    ) - 3
    y = draw_paragraph(
        pdf,
        f"Reproducibility: seed 2027; scikit-learn {sklearn.__version__}; "
        f"NumPy {np.__version__}; pandas {pd.__version__}; Python "
        f"{platform.python_version()}. Detailed folds and results: "
        f"{Path(csv_path).name}; environment: {Path(environment_path).name}.",
        margin,
        y,
        content_width,
        small,
    )
    if y < 28:
        raise RuntimeError(f"One-page report overflowed the page (bottom y={y:.1f}).")
    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor("#62676D"))
    pdf.drawRightString(page_width - margin, 22, "Analysis summary - 11-point text")
    pdf.showPage()

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(margin, page_height - margin - 10, "Appendix: Author-written SVM skill")
    pdf.setFont("SkillAppendix", 8.5)
    y = page_height - margin - 34
    line_height = 11
    max_chars = int(content_width / stringWidth("M", "SkillAppendix", 8.5))
    for original_line in skill_text.splitlines():
        wrapped = textwrap.wrap(
            original_line,
            width=max_chars,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        for line in wrapped:
            if y < margin:
                pdf.showPage()
                pdf.setFont("SkillAppendix", 8.5)
                y = page_height - margin
            pdf.drawString(margin, y, line)
            y -= line_height
    pdf.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_csv", type=Path)
    parser.add_argument(
        "--skill",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    skill_text = args.skill.read_text(encoding="utf-8")
    data = pd.read_csv(args.data_csv)
    required = FEATURES + ["chd"]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")
    if data[required].isna().any().any():
        raise ValueError("Missing values found; define and document imputation first.")
    if set(data["famhist"].unique()) - {"Absent", "Present"}:
        raise ValueError("Unexpected famhist values; expected Absent and Present.")
    if set(data["chd"].unique()) - {0, 1}:
        raise ValueError("Expected binary chd labels coded 0 and 1.")
    X = data[FEATURES].copy()
    X["famhist"] = X["famhist"].map({"Absent": 1, "Present": 2})
    X = X.astype(float)
    y = data["chd"].astype(int)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite predictor values found.")
    class_counts = y.value_counts().sort_index().to_dict()
    if min(class_counts.values()) < 10:
        raise ValueError("10-fold stratified CV requires at least 10 cases per class.")

    outer_cv = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=args.seed
    )
    oof_scores = {family: np.full(len(y), np.nan) for family in FAMILY_ORDER}
    selected_oof = np.full(len(y), np.nan)
    fold_records = []
    importance_records = []
    selected_family_records = []

    for fold_number, (train_index, test_index) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        searches = {}
        for family in FAMILY_ORDER:
            search = fit_search(
                family, X_train, y_train, args.seed + 100 * fold_number
            )
            searches[family] = search
            estimator = search.best_estimator_
            scores = estimator.decision_function(X_test)
            predictions = estimator.predict(X_test)
            oof_scores[family][test_index] = scores
            fold_records.append(
                {
                    "fold": fold_number,
                    "kernel": family,
                    "roc_auc": roc_auc_score(y_test, scores),
                    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                    "support_vectors": support_count(estimator),
                    "best_inner_cv_roc_auc": search.best_score_,
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                }
            )

        selected_family, best_family, one_se = select_with_one_se(searches)
        selected_estimator = searches[selected_family].best_estimator_
        selected_scores = selected_estimator.decision_function(X_test)
        selected_predictions = selected_estimator.predict(X_test)
        selected_oof[test_index] = selected_scores
        selected_family_records.append(
            {
                "fold": fold_number,
                "selected_kernel": selected_family,
                "highest_inner_score_kernel": best_family,
                "one_se": one_se,
                "roc_auc": roc_auc_score(y_test, selected_scores),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, selected_predictions
                ),
                "support_vectors": support_count(selected_estimator),
                "best_params": json.dumps(
                    searches[selected_family].best_params_, sort_keys=True
                ),
            }
        )
        importance = permutation_importance(
            selected_estimator,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=10,
            random_state=args.seed + fold_number,
            n_jobs=1,
        )
        for feature, mean, sd in zip(
            FEATURES, importance.importances_mean, importance.importances_std
        ):
            importance_records.append(
                {
                    "fold": fold_number,
                    "feature": feature,
                    "mean_roc_auc_decrease": mean,
                    "permutation_sd": sd,
                    "selected_kernel": selected_family,
                }
            )
        print(
            f"Completed outer fold {fold_number}/10; selected {selected_family}.",
            flush=True,
        )

    fold_results = pd.DataFrame(fold_records)
    fold_results.to_csv(output / "outer-cv-fold-results.csv", index=False)
    selected_results = pd.DataFrame(selected_family_records)
    selected_results.to_csv(output / "selected-procedure-fold-results.csv", index=False)

    summary_rows = []
    for family in FAMILY_ORDER:
        subset = fold_results[fold_results["kernel"] == family]
        summary_rows.append(
            {
                "kernel": family,
                "roc_auc_mean": subset["roc_auc"].mean(),
                "roc_auc_sd": subset["roc_auc"].std(ddof=1),
                "balanced_accuracy_mean": subset["balanced_accuracy"].mean(),
                "balanced_accuracy_sd": subset["balanced_accuracy"].std(ddof=1),
                "support_vectors_mean": subset["support_vectors"].mean(),
                "support_vectors_sd": subset["support_vectors"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("kernel")
    summary.to_csv(output / "kernel-cv-summary.csv")

    importance_df = pd.DataFrame(importance_records)
    importance_summary = (
        importance_df.groupby("feature")["mean_roc_auc_decrease"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .rename(
            columns={
                "mean": "mean_roc_auc_decrease",
                "std": "between_fold_sd",
            }
        )
    )
    importance_summary.to_csv(output / "permutation-feature-importance.csv")
    importance_df.to_csv(output / "permutation-feature-importance-by-fold.csv", index=False)

    final_searches = {
        family: fit_search(family, X, y, args.seed + 10000)
        for family in FAMILY_ORDER
    }
    selected_family, _, _ = select_with_one_se(final_searches)
    final_search = final_searches[selected_family]
    final_model = final_search.best_estimator_
    final_sv = support_count(final_model)
    joblib.dump(final_model, output / "selected-svm-model.joblib")
    final_params = final_search.best_params_
    (output / "selected-model.json").write_text(
        json.dumps(
            {
                "selected_kernel": selected_family,
                "best_params": final_params,
                "support_vectors": final_sv,
                "n_observations": len(y),
                "support_vector_ratio": final_sv / len(y),
                "seed": args.seed,
                "selection_rule": (
                    "Among kernel families whose best 10-fold inner-CV ROC-AUC is "
                    "within one standard error of the highest, prefer linear, then "
                    "polynomial, then RBF."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_roc_grid(
        oof_scores,
        y,
        selected_oof,
        output / "nested-cv-roc-grid.png",
    )
    importance_top = importance_summary.head(3)
    top_features = "; ".join(
        f"{feature} ({row['mean_roc_auc_decrease']:.3f})"
        for feature, row in importance_top.iterrows()
    )
    env_text = (
        f"Python: {platform.python_version()}\n"
        f"Platform: {platform.platform()}\n"
        f"scikit-learn: {sklearn.__version__}\n"
        f"NumPy: {np.__version__}\n"
        f"pandas: {pd.__version__}\n"
        f"Matplotlib: {matplotlib.__version__}\n"
        f"ReportLab: {reportlab.Version}\n"
        "SVM: sklearn.svm.SVC; scaling: sklearn.preprocessing.StandardScaler\n"
        f"Seed: {args.seed}\n"
        f"Input: {args.data_csv.resolve()}\n"
        f"Skill: {args.skill.resolve()}\n"
        f"Command: {sys.executable} {Path(__file__).resolve()} "
        f"{args.data_csv.resolve()} --skill {args.skill.resolve()} "
        f"--seed {args.seed}\n"
        f"Rows: {len(y)}; class counts: {class_counts}\n"
        f"Predictors: {FEATURES}; famhist mapping: Absent=1, Present=2\n"
        "CV: nested stratified 10-fold; inner metric: ROC-AUC; "
        "permutation importance: 10 repeats per outer fold.\n"
    )
    environment_path = output / "environment.txt"
    environment_path.write_text(env_text, encoding="utf-8")
    selected_auc = selected_results["roc_auc"]
    selected_ba = selected_results["balanced_accuracy"]
    report_values = {
        "summary": summary,
        "selected_family": selected_family,
        "selected_params": final_params,
        "full_sv": final_sv,
        "full_sv_ratio": final_sv / len(y),
        "selected_auc_mean": selected_auc.mean(),
        "selected_auc_sd": selected_auc.std(ddof=1),
        "selected_ba_mean": selected_ba.mean(),
        "selected_ba_sd": selected_ba.std(ddof=1),
        "top_features": top_features,
    }
    draw_report(
        output / "svm-analysis-report.pdf",
        report_values,
        output / "nested-cv-roc-grid.png",
        skill_text,
        output / "kernel-cv-summary.csv",
        environment_path,
    )
    (output / "svm-analysis-summary.md").write_text(
        "# SVM Analysis Summary\n\n"
        f"- Selected kernel: {selected_family}\n"
        f"- Parameters: `{json.dumps(final_params, sort_keys=True)}`\n"
        f"- Nested-CV ROC-AUC: {selected_auc.mean():.3f} +/- "
        f"{selected_auc.std(ddof=1):.3f}\n"
        f"- Nested-CV balanced accuracy: {selected_ba.mean():.3f} +/- "
        f"{selected_ba.std(ddof=1):.3f}\n"
        f"- Support vectors: {final_sv}/{len(y)} ({final_sv / len(y):.1%})\n"
        f"- Top held-out permutation importance: {top_features}\n\n"
        "The PDF contains the one-page report followed by the skill text exactly "
        "as read from the supplied `1.md` file.\n",
        encoding="utf-8",
    )
    print(f"Results saved to {output}")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(selected_results.to_string(index=False))
    print("Selected full-data model:", selected_family, final_params)
    print("Top permutation importance:\n", importance_summary.head(5))


if __name__ == "__main__":
    main()
