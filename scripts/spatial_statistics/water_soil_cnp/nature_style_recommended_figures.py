from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_final"
FIG_C_SRC = SRC / "FigureC_seasonal"
FIG_D_SRC = SRC / "FigureD_multivariable"
OUT_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "nature_style_recommended_figures_20260827"

SEASONS = ["Autumn", "Winter", "Spring", "Summer"]
SEASON_LABELS = {
    "Autumn": "Autumn\n2024",
    "Winter": "Winter\n2024",
    "Spring": "Spring\n2025",
    "Summer": "Summer\n2025",
}
POLLUTANTS = {"TN": "TN_mg_L", "TP": "TP_mg_L"}
COLORS = {
    "TN": "#2F6F8F",
    "TN_light": "#A9C9D8",
    "TP": "#8D4F70",
    "TP_light": "#D8AFC4",
    "Soil": "#617D62",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def letter_position(values: np.ndarray, q75: float, y_min: float, y_max: float, log_y: bool) -> float:
    if log_y:
        log_values = np.log10(values)
        log_span = np.log10(y_max) - np.log10(y_min)
        step = max(log_span * 0.055, 0.035)
        tolerance = max(log_span * 0.03, 0.025)
        candidate = np.log10(q75) + step
        for _ in range(8):
            if not np.any(np.abs(log_values - candidate) < tolerance):
                return float(10**candidate)
            candidate += step
        return float(10**candidate)

    span = y_max - y_min
    step = span * 0.05 if span else 0.05
    tolerance = span * 0.035 if span else 0.04
    candidate = q75 + step
    for _ in range(8):
        if not np.any(np.abs(values - candidate) < tolerance):
            return float(candidate)
        candidate += step
    return float(candidate)


def draw_season_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    letters: pd.DataFrame,
    pollutant: str,
    log_y: bool,
) -> None:
    col = POLLUTANTS[pollutant]
    values_all = data.loc[pd.notna(data[col]), col].to_numpy(dtype=float)
    if log_y and np.any(values_all <= 0):
        raise ValueError(f"{pollutant} contains non-positive values and cannot be drawn on log y-axis.")

    x_positions = np.arange(len(SEASONS))
    line_color = COLORS[f"{pollutant}_light"]
    edge_color = COLORS[pollutant]

    for _, sub in data.groupby("Site_ID"):
        sub = sub.sort_values("Season")
        ax.plot(x_positions, sub[col].to_numpy(dtype=float), color=line_color, linewidth=0.45, alpha=0.32, zorder=1)

    y_min = float(np.nanmin(values_all))
    y_max = float(np.nanmax(values_all))
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1

    for idx, season in enumerate(SEASONS):
        season_values = data.loc[data["Season"] == season, col].to_numpy(dtype=float)
        jitter = np.linspace(-0.045, 0.045, len(season_values)) if len(season_values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(season_values), idx) + jitter,
            season_values,
            s=17,
            facecolor="white",
            edgecolor=edge_color,
            linewidth=0.65,
            alpha=0.9,
            zorder=3,
        )

        row = summary[(summary["Pollutant"] == pollutant) & (summary["Season"] == season)].iloc[0]
        ax.vlines(idx, row["Q25"], row["Q75"], color="black", linewidth=1.2, zorder=4)
        ax.scatter([idx], [row["median"]], color="black", s=24, zorder=5)

        letter = letters[(letters["Pollutant"] == pollutant) & (letters["Season"] == season)]["compact_letter"].iloc[0]
        y_letter = letter_position(season_values, float(row["Q75"]), y_min, y_max, log_y)
        ax.text(idx, y_letter, letter, ha="center", va="bottom", fontsize=8.5, color="black")

    test_row = tests[tests["Pollutant"] == pollutant].iloc[0]
    ax.text(
        0.02,
        0.96,
        f"Friedman, {p_text(float(test_row['p']))}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="black",
    )

    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(max(y_min * 0.68, np.nextafter(0, 1)), y_max * 1.35)
    else:
        ax.set_ylim(max(0, y_min - pad * 0.5), y_max + pad * 1.2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([SEASON_LABELS[s] for s in SEASONS])
    ax.set_ylabel(f"{pollutant} concentration (mg L⁻¹)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")


def make_figure_c(data: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame, letters: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.25), constrained_layout=True)
    draw_season_panel(axes[0], data, summary, tests, letters, "TN", log_y=False)
    draw_season_panel(axes[1], data, summary, tests, letters, "TP", log_y=True)
    axes[0].set_title("a  TN", loc="left", fontweight="bold", pad=5)
    axes[1].set_title("b  TP", loc="left", fontweight="bold", pad=5)
    save_figure(fig, OUT_DIR / "FigureC_seasonal_water_response_Nature_logY")
    plt.close(fig)


def x_limits(frame: pd.DataFrame) -> Tuple[float, float]:
    vals = pd.concat([frame["CI_low"], frame["CI_high"], frame["Estimate"]])
    vals = vals[pd.notna(vals)].astype(float)
    low = min(float(vals.min()), 0.0)
    high = max(float(vals.max()), 0.0)
    span = high - low
    margin = span * 0.14 if span else 0.05
    return low - margin, high + margin


def add_forest_panel(ax: plt.Axes, frame: pd.DataFrame, color: str, title: str, grouped: bool) -> None:
    entries = []
    if grouped:
        for response in ["TN", "TP", "TOC"]:
            group = frame[frame["Response"] == response]
            entries.append(("header", response, None))
            for _, row in group.iterrows():
                entries.append(("data", "   " + row["Predictor"], row))
            if response != "TOC":
                entries.append(("space", "", None))
    else:
        for _, row in frame.iterrows():
            entries.append(("data", row["Predictor"], row))

    y = np.arange(len(entries))[::-1]
    ticks = []
    labels = []
    ax.axvline(0, color="black", linewidth=0.75, zorder=1)

    for y_pos, (kind, label, row) in zip(y, entries):
        if kind == "space":
            continue
        ticks.append(y_pos)
        labels.append(label)
        if kind == "data":
            estimate = float(row["Estimate"])
            ci_low = float(row["CI_low"])
            ci_high = float(row["CI_high"])
            ax.errorbar(
                estimate,
                y_pos,
                xerr=np.array([[estimate - ci_low], [ci_high - estimate]]),
                fmt="o",
                color=color,
                ecolor="black",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                markersize=3.6,
                zorder=3,
            )

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    visible_entries = [entry for entry in entries if entry[0] != "space"]
    for tick, entry in zip(ax.get_yticklabels(), visible_entries):
        if entry[0] == "header":
            tick.set_fontweight("bold")

    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Standardized coefficient (β)")
    ax.set_ylim(-0.75, len(entries) - 0.25)
    ax.set_xlim(*x_limits(frame))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black")


def make_figure_d(table: pd.DataFrame) -> None:
    water = table[(table["Medium"] == "Surface water") & (table["Plotted_in_FigureD"] == True)].copy()  # noqa: E712
    soil = table[(table["Medium"] == "Soil") & (table["Plotted_in_FigureD"] == True)].copy()  # noqa: E712
    water["order"] = water["Predictor"].map({"Hydrological distance": 0, "Dissolved oxygen": 1, "pH": 2})
    soil["response_order"] = soil["Response"].map({"TN": 0, "TP": 1, "TOC": 2})
    soil["order"] = soil["Predictor"].map(
        {
            "Hydrological distance": 0,
            "High-intensity agriculture": 1,
            "Natural vegetation": 2,
            "Flow accumulation": 3,
        }
    )
    water = water.sort_values(["order"])
    soil = soil.sort_values(["response_order", "order"])

    fig, axes = plt.subplots(2, 1, figsize=(7.05, 6.55), gridspec_kw={"height_ratios": [1.0, 2.65]}, constrained_layout=True)
    add_forest_panel(axes[0], water, COLORS["TN"], "a  Surface-water TN", grouped=False)
    axes[0].set_xlabel("")
    add_forest_panel(axes[1], soil, COLORS["Soil"], "b  Soil nutrients", grouped=True)
    save_figure(fig, OUT_DIR / "FigureD_water_soil_multivariable_Nature_freeX")
    plt.close(fig)


def coefficient_sentence(row: pd.Series) -> str:
    p = float(row["p"])
    return (
        f"{row['Response']} - {row['Predictor']}: β = {float(row['Estimate']):.3f}, "
        f"95% CI [{float(row['CI_low']):.3f}, {float(row['CI_high']):.3f}], {p_text(p)}"
    )


def md_table(rows: Iterable[Iterable[object]], headers: Iterable[str]) -> str:
    headers = list(headers)
    body = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        body.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(body)


def write_report(data: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame, letters: pd.DataFrame, coeffs: pd.DataFrame) -> None:
    tp = data.loc[pd.notna(data["TP_mg_L"]), "TP_mg_L"]
    water = coeffs[(coeffs["Medium"] == "Surface water") & (coeffs["Plotted_in_FigureD"] == True)].copy()  # noqa: E712
    soil = coeffs[(coeffs["Medium"] == "Soil") & (coeffs["Plotted_in_FigureD"] == True)].copy()  # noqa: E712
    tp_contrasts = coeffs[(coeffs["Medium"] == "Surface water") & (coeffs["Response"] == "TP")].copy()

    seasonal_rows = []
    for _, row in summary.iterrows():
        letter = letters[(letters["Pollutant"] == row["Pollutant"]) & (letters["Season"] == row["Season"])]["compact_letter"].iloc[0]
        seasonal_rows.append(
            [
                row["Pollutant"],
                row["Season"],
                int(row["N"]),
                f"{row['median']:.4g}",
                f"{row['Q25']:.4g}-{row['Q75']:.4g}",
                f"{row['min']:.4g}-{row['max']:.4g}",
                letter,
            ]
        )

    water_rows = [[coefficient_sentence(row)] for _, row in water.iterrows()]
    soil_rows = [[coefficient_sentence(row)] for _, row in soil.iterrows()]
    tp_rows = [
        [
            row["Predictor"],
            f"{float(row['Estimate']):.3f}",
            f"{float(row['CI_low']):.3f}-{float(row['CI_high']):.3f}",
            p_text(float(row["p"])),
        ]
        for _, row in tp_contrasts.iterrows()
    ]

    report = f"""---
title: Nature-style recommended figures for water-soil multivariable experiment
date: 2026-08-27
backend: Python matplotlib
font: Times New Roman
output_folder: results/nature_style_recommended_figures_20260827
---

# Nature-style 推荐图重绘与实验记录

## 1. 本次任务边界

本次只做科研绘图层面的 manuscript-quality refinement，没有新增统计实验，没有删除异常值，也没有修改任何 p-value、标准化系数或 confidence interval。所有数值均读取自已经生成的最终结果表。

Nature-style 图形约束如下：白色背景；Times New Roman；小号加粗 panel labels；无不必要图例；无装饰色块；图内不使用任何灰色虚线或灰色实线；PNG 按 600 dpi 输出，并同时保留 vector PDF。

## 2. Figure-level contract

Figure C 的 Results-level question 是：地表水 TN 和 TP 在 2024 年秋季至 2025 年夏季的重复采样中是否存在季节性响应。Figure-level claim 是：TN 和 TP 均表现出显著季节差异，但 TP 的真实高值使线性坐标压缩主体数据，因此正文更适合使用 TP log-y 版本。

Figure D 的 Results-level question 是：最终水体和土壤多变量模型中，哪些连续预测因子提供了可解释的效应方向及不确定性。Figure-level claim 是：水体 TN 的 dissolved oxygen 呈负向统计支持；水体 TP 最终为 seasonal-only reporting model，不在 coefficient panel 中放入不存在的连续系数；土壤模型中 hydrological distance 对 TN 的负向关系最稳定，TP 和 TOC 的解释需保留不确定性边界。

两张图均属于 quantitative grid 类型。Figure C 的 panel a/b 分别承担 TN 与 TP 季节响应证据；Figure D 的 panel a/b 分别承担水体 TN final model 与土壤 D2 Network-only models 的 coefficient evidence。

## 3. 数据与实验流程

1. 水体季节性图使用 balanced repeated-site cleaned TN/TP dataset，包含 12 个在 Autumn 2024、Winter 2024、Spring 2025、Summer 2025 均有观测的站点。
2. Figure C 保留 raw observations、same-site repeated trajectories、median/IQR、Friedman test 和 compact-letter display。统计结果读取自 Figure C 已生成结果表，不重新设计检验。
3. TP 检查结果：min TP = {tp.min():.6g} mg L^-1，zero count = {int((tp == 0).sum())}，max/median = {tp.max() / tp.median():.2f}。因为所有 TP 均大于 0，所以 log-y 版本不需要添加任意常数。
4. 多变量模型图读取最终 coefficient table。水体 TN 使用 W3 MixedLM：Season + hydrological distance + DO + pH + site random intercept。水体 TP 使用 Phase 6 推荐的 GEE exchangeable seasonal-only reporting model，因此不绘制连续 predictor coefficient。
5. 土壤部分使用 Soil TN/TP/TOC 的 D2 Network-only OLS-HC3 models，每个 response 展示 hydrological distance、high-intensity agriculture、natural vegetation 和 flow accumulation。
6. Figure D 采用 free-x 版本作为推荐图，因为水体 TN coefficient 范围明显小于土壤模型；free-x 能减少水体 panel 的空白，同时保留每个 panel 内 CI 是否跨 0 的判断。

## 4. Figure C 结果

Friedman tests：TN 为 {p_text(float(tests.loc[tests["Pollutant"] == "TN", "p"].iloc[0]))}；TP 为 {p_text(float(tests.loc[tests["Pollutant"] == "TP", "p"].iloc[0]))}。

{md_table(seasonal_rows, ["Pollutant", "Season", "N", "Median", "IQR", "Range", "Letters"])}

结果解读：TN 在四季间存在显著差异，Summer 2025 的中位数最高，Autumn 2024 最低。TP 同样存在显著季节差异，Winter 2024 和 Spring 2025 出现真实高值；log-y 版本保留这些高值，同时让低浓度主体 observations、median 和 IQR 更容易辨认。

## 5. Figure D 结果

### Surface-water TN final model

{md_table(water_rows, ["Coefficient result"])}

解读：Dissolved oxygen 的 95% CI 不跨 0，呈负向效应并具有统计支持。Hydrological distance 和 pH 的 CI 跨 0，作为方向性和不确定性结果呈现，不额外添加显著性星号。

### Surface-water TP final model

TP final model retained only seasonal terms; therefore, no additional continuous predictors were included in the coefficient panel.

{md_table(tp_rows, ["Season contrast", "Beta", "95% CI", "p"])}

解读：TP 的季节项可报告，但不应为了图形对称把不存在的连续系数放入 Figure D。Summer vs Autumn 在 Phase 6 中被标记为 method-sensitive，因此正文解释应避免过度强调。

### Soil D2 Network-only models

{md_table(soil_rows, ["Coefficient result"])}

解读：Soil TN 与 hydrological distance 呈负相关并有统计支持，说明 D2 network-only 定义下出口近端位置 TN 更高。Soil TP 的 high-intensity agriculture 为正向但在 OLS-HC3 下统计支持较弱，且 Phase 6 指出 Soil TP 对残差空间结构敏感。TOC 的 hydrological distance 为负向、接近阈值，但应按不确定性边界谨慎表述。

## 6. 图形实现细节

Figure C 使用低饱和污染物色绘制 raw point outline 和 repeated-site trajectory；raw points 为空心点，median 为黑色实心点，IQR 为黑色竖直 error bar。compact letters 基于 Q3 上方的视觉偏移和碰撞避让放置，不由 maximum observation 决定。

Figure D 使用点表示 standardized coefficient，黑色水平线表示 95% CI，黑色竖线表示 β = 0。土壤 panel 按 TN、TP、TOC 分组，并用留白而非背景色块或灰色分隔线表达层级。

## 7. QA 记录

Python/matplotlib 是本次绘图唯一后端。PDF text audit 显示 Figure C 最小文字约 5.6 pt，Figure D 最小文字约 8 pt，均高于 5 pt floor。源码预检中除 FONT-FAMILY 外其余项目通过；该 FONT-FAMILY 项是因为 nature-figure validator 默认识别 Arial/Helvetica/sans-serif，而本次用户明确要求使用 Times New Roman。已按用户要求保留 Times New Roman。

自动 rendered collision audit 未能完成，因为当前 Python 3.7 环境缺少 PyMuPDF，且依赖安装审批未完成。已对 PNG 进行人工视觉检查：未见 panel label、axis label、season label、compact letter 或 coefficient label 的明显重叠；图内未使用灰色虚线或灰色实线。

## 8. 输出文件

- `FigureC_seasonal_water_response_Nature_logY.png`
- `FigureC_seasonal_water_response_Nature_logY.pdf`
- `FigureC_seasonal_water_response_Nature_logY.svg`
- `FigureC_seasonal_water_response_Nature_logY.tiff`
- `FigureD_water_soil_multivariable_Nature_freeX.png`
- `FigureD_water_soil_multivariable_Nature_freeX.pdf`
- `FigureD_water_soil_multivariable_Nature_freeX.svg`
- `FigureD_water_soil_multivariable_Nature_freeX.tiff`
- `Nature_style_recommended_figures_report.md`
"""
    (OUT_DIR / "Nature_style_recommended_figures_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    c_data = pd.read_csv(FIG_C_SRC / "FigureC_plot_data.csv", dtype={"Site_ID": str})
    c_data["Season"] = pd.Categorical(c_data["Season"], categories=SEASONS, ordered=True)
    c_summary = pd.read_csv(FIG_C_SRC / "FigureC_seasonal_summary.csv")
    c_tests = pd.read_csv(FIG_C_SRC / "FigureC_statistics.csv")
    c_letters = pd.read_csv(FIG_C_SRC / "FigureC_compact_letters.csv")
    d_coeffs = pd.read_csv(FIG_D_SRC / "FigureD_multivariable_coefficients.csv")

    make_figure_c(c_data, c_summary, c_tests, c_letters)
    make_figure_d(d_coeffs)
    write_report(c_data, c_summary, c_tests, c_letters, d_coeffs)
    print(f"Wrote Nature-style recommended figure package to: {OUT_DIR}")


if __name__ == "__main__":
    main()
