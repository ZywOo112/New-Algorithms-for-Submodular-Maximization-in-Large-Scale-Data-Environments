#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果可视化脚本
============================================================
根据实验数据 (1.docx) 生成论文中的对比图表。

图表包括:
  fig1: 不同算法影响力传播范围对比 (论文图 5.1)
  fig2: 精确 Oracle 调用次数对比 (论文图 5.2)
  fig3: 总传播模拟预算对比 (论文图 5.3)
  fig4: 运行时间对比 (论文图 5.4)
  fig5: 不同流顺序下的传播范围分布 (论文图 5.5)
  fig_scalability: 可扩展性实验 (运行时间 & 传播范围 vs 数据集规模)

数据来源: 1.docx (加速模拟计算结果)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import os

# ============================================================================
# Section 0: 中文字体配置
# ============================================================================

def setup_chinese_font():
    """配置 matplotlib 使用系统中可用的中文字体"""
    candidate_fonts = ['SimHei', 'Microsoft YaHei', 'STXihei', 'SimSun', 'KaiTi']
    for font_name in candidate_fonts:
        try:
            fp = FontProperties(family=font_name)
            font_path = findfont(fp, fallback_to_default=False)
            if font_path:
                matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"使用中文字体: {font_name}")
                return
        except Exception:
            continue
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("未找到中文字体，使用默认字体")

setup_chinese_font()

# ============================================================================
# Section 1: 实验数据 (来源: 1.docx)
# ============================================================================

# 算法名称及其简写
ALGO_SHORT = {
    "ATB-Max":                "ATB-Max",
    "ATB-Max (无估计器)":      "ATB-Max\n(w/o Est)",
    "ATB-Max (无UCB)":        "ATB-Max\n(w/o UCB)",
    "固定阈值":                "Fixed-Thr",
    "随机阈值":                "Rand-Thr",
    "Sieve-Streaming":        "Sieve-Str",
    "Greedy":                 "Greedy",
}

ALGO_COLORS = {
    "ATB-Max":                "#2196F3",
    "ATB-Max (无估计器)":      "#FF9800",
    "ATB-Max (无UCB)":        "#9E9E9E",
    "固定阈值":                "#607D8B",
    "随机阈值":                "#795548",
    "Sieve-Streaming":        "#4CAF50",
    "Greedy":                 "#F44336",
}

ALGO_ORDER = [
    "ATB-Max",
    "ATB-Max (无估计器)",
    "ATB-Max (无UCB)",
    "固定阈值",
    "随机阈值",
    "Sieve-Streaming",
    "Greedy",
]

# ---- 传播范围数据 (10次重复) ----
spread_data = {
    "ATB-Max":                [381.14, 379.76, 376.98, 371.00, 379.17, 381.56, 381.42, 378.93, 380.39, 382.70],
    "ATB-Max (无估计器)":      [386.29, 388.25, 387.79, 387.38, 386.87, 387.42, 389.21, 385.64, 387.32, 384.90],
    "ATB-Max (无UCB)":        [334.83, 348.66, 355.74, 371.82, 331.54, 346.64, 359.59, 368.39, 363.55, 375.54],
    "固定阈值":                [360.90, 367.13, 347.33, 371.67, 354.51, 372.89, 342.30, 359.20, 349.33, 368.15],
    "随机阈值":                [376.29, 360.36, 363.31, 385.81, 391.38, 374.81, 388.58, 375.42, 383.33, 379.71],
    "Sieve-Streaming":        [384.36, 375.11, 380.08, 385.35, 386.99, 387.65, 370.25, 375.01, 376.16, 370.75],
    "Greedy":                 [389.0],  # 单值 (离线算法)
}

# ---- 精确 Oracle 调用次数 (单次运行总值) ----
oracle_calls = {
    "ATB-Max":                16548,
    "ATB-Max (无估计器)":      29702,
    "ATB-Max (无UCB)":        19561,
    "固定阈值":                19561,
    "随机阈值":                17737,
    "Sieve-Streaming":        62423,
    "Greedy":                 26896,
}

# ---- 总传播模拟预算 ----
total_budget = {
    "ATB-Max":                5.45e6,
    "ATB-Max (无估计器)":      8.40e6,
    "ATB-Max (无UCB)":        6.45e6,
    "固定阈值":                6.45e6,
    "随机阈值":                5.85e6,
    "Sieve-Streaming":        1.56e8,
    "Greedy":                 7.05e6,
}

# ---- 运行时间 (秒) ----
runtime = {
    "ATB-Max":                646,
    "ATB-Max (无估计器)":      1846,
    "ATB-Max (无UCB)":        921,
    "固定阈值":                920,
    "随机阈值":                664,
    "Sieve-Streaming":        3528,
    "Greedy":                 None,  # 未提供
}

# ---- 可扩展实验数据 ----
# ATB-Max 在不同数据集上的运行时间和传播范围
scalability_data = {
    "medium-scale\n(n=2000, BA)":     {"nodes": 2000,  "runtime": 249,  "spread": 212.03},
    "random\n(n=3000, ER)":           {"nodes": 3000,  "runtime": 365,  "spread": 174.06},
    "small-world\n(n=3000, WS)":      {"nodes": 3000,  "runtime": 423,  "spread": 245.28},
    "facebook-scale\n(n=4039, BA)":   {"nodes": 4039,  "runtime": 646,  "spread": 377.33},
    "large-scale\n(n=8000, BA)":      {"nodes": 8000,  "runtime": 1267, "spread": 476.20},
}


# ============================================================================
# Section 2: 辅助函数
# ============================================================================

def compute_spread_stats():
    """计算传播范围的均值和标准差"""
    stats = {}
    for name in ALGO_ORDER:
        data = spread_data.get(name, [])
        if data:
            stats[name] = {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1) if len(data) > 1 else 0,
                'min': np.min(data),
                'max': np.max(data),
            }
    return stats


def get_color(name):
    return ALGO_COLORS.get(name, '#999999')


def get_short_name(name):
    return ALGO_SHORT.get(name, name)


# ============================================================================
# Section 3: 图表生成
# ============================================================================

def fig1_spread_comparison(output_dir="figures"):
    """不同算法的影响力传播范围对比"""
    stats = compute_spread_stats()

    names = [a for a in ALGO_ORDER if a in stats]
    means = [stats[a]['mean'] for a in names]
    stds = [stats[a]['std'] for a in names]
    colors = [get_color(a) for a in names]
    labels = [get_short_name(a) for a in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                  edgecolor='white', linewidth=0.8, width=0.65)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('影响力传播范围', fontsize=12)
    ax.set_title('不同算法的影响力传播范围对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'fig5.1_spread_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  影响力传播范围对比图已保存")


def fig2_oracle_calls(output_dir="figures"):
    """精确 Oracle 调用次数对比"""
    names = [a for a in ALGO_ORDER if a in oracle_calls]
    vals = [oracle_calls[a] for a in names]
    colors = [get_color(a) for a in names]
    labels = [get_short_name(a) for a in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8, width=0.65)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('精确 Oracle 调用次数', fontsize=12)
    ax.set_title('不同算法的精确 Oracle 调用次数对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'fig5.2_oracle_calls.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  精确 Oracle 调用次数对比图已保存")


def fig3_total_budget(output_dir="figures"):
    """总传播模拟预算对比"""
    names = [a for a in ALGO_ORDER if a in total_budget]
    vals = [total_budget[a] for a in names]
    colors = [get_color(a) for a in names]
    labels = [get_short_name(a) for a in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8, width=0.65)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f'{val:.2e}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('总传播模拟预算', fontsize=12)
    ax.set_title('不同算法的总传播模拟预算对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'fig5.3_total_budget.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  总传播模拟预算对比图已保存")


def fig4_runtime(output_dir="figures"):
    """运行时间对比"""
    names = [a for a in ALGO_ORDER if a in runtime and runtime[a] is not None]
    vals = [runtime[a] for a in names]
    colors = [get_color(a) for a in names]
    labels = [get_short_name(a) for a in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8, width=0.65)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f'{val}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('运行时间 / 秒', fontsize=12)
    ax.set_title('不同算法的运行时间对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'fig5.4_runtime.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  运行时间对比图已保存")


def fig5_spread_distribution(output_dir="figures"):
    """不同流顺序下的传播范围分布 (箱线图)"""
    # 排除 Greedy (只有一个值)
    box_names = [a for a in ALGO_ORDER if a in spread_data and len(spread_data[a]) > 1]
    box_data = [spread_data[a] for a in box_names]
    labels = [get_short_name(a) for a in box_names]
    colors = [get_color(a) for a in box_names]

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    widths=0.55, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    for i, data in enumerate(box_data):
        jitter = np.random.normal(0, 0.04, size=len(data))
        ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data,
                   alpha=0.5, s=20, color='black', zorder=5)

    if "Greedy" in spread_data:
        greedy_val = spread_data["Greedy"][0]
        ax.axhline(y=greedy_val, color=ALGO_COLORS["Greedy"], linestyle='--',
                   linewidth=1.2, alpha=0.7, label=f'Greedy ({greedy_val:.0f})')
        ax.legend(fontsize=8, loc='lower right')

    ax.set_ylabel('影响力传播范围', fontsize=12)
    ax.set_title('不同流顺序下的传播范围分布', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'fig5.5_spread_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  传播范围分布图已保存")


def fig_scalability(output_dir="figures"):
    """可扩展性实验：ATB-Max 在不同规模数据集上的表现"""
    ds_sorted = sorted(scalability_data.items(), key=lambda x: x[1]['nodes'])

    ds_names = [name for name, _ in ds_sorted]
    ds_nodes = [d['nodes'] for _, d in ds_sorted]
    ds_runtime = [d['runtime'] for _, d in ds_sorted]
    ds_spread = [d['spread'] for _, d in ds_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 运行时间 vs 数据集规模
    ax1.plot(ds_nodes, ds_runtime, 'o-', color='#2196F3', linewidth=2,
             markersize=10, markerfacecolor='white', markeredgewidth=2)
    for nx, rt, name in zip(ds_nodes, ds_runtime, ds_names):
        ax1.annotate(f'{rt}s', (nx, rt), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('节点数', fontsize=12)
    ax1.set_ylabel('运行时间 / 秒', fontsize=12)
    ax1.set_title('(a) 运行时间随数据集规模的变化', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # (b) 传播范围 vs 数据集规模
    ax2.plot(ds_nodes, ds_spread, 's-', color='#FF5722', linewidth=2,
             markersize=10, markerfacecolor='white', markeredgewidth=2)
    for nx, sp, name in zip(ds_nodes, ds_spread, ds_names):
        ax2.annotate(f'{sp:.1f}', (nx, sp), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('节点数', fontsize=12)
    ax2.set_ylabel('影响力传播范围', fontsize=12)
    ax2.set_title('(b) 传播范围随数据集规模的变化', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('ATB-Max 在不同规模数据集上的可扩展性',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'fig_scalability.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  可扩展性图已保存")


# ============================================================================
# Section 4: 主入口
# ============================================================================

def main():
    output_dir = "figures"
    print("开始生成实验图表...\n")

    fig1_spread_comparison(output_dir)
    fig2_oracle_calls(output_dir)
    fig3_total_budget(output_dir)
    fig4_runtime(output_dir)
    fig5_spread_distribution(output_dir)
    fig_scalability(output_dir)

    # 打印统计摘要
    print(f"\n{'=' * 60}")
    print("传播范围统计摘要")
    print(f"{'=' * 60}")
    stats = compute_spread_stats()
    greedy_mean = stats.get("Greedy", {}).get("mean", 389.0)
    print(f"{'算法':<24} {'均值':>8} {'标准差':>6} {'Ratio/Greedy':>13}")
    print("-" * 60)
    for name in ALGO_ORDER:
        if name in stats:
            s = stats[name]
            ratio = s['mean'] / max(greedy_mean, 0.01)
            print(f"{name:<24} {s['mean']:>8.2f} {s['std']:>6.2f} {ratio:>12.4f}")

    print(f"\n所有图表已保存至 {output_dir}/ 目录")


if __name__ == "__main__":
    main()
