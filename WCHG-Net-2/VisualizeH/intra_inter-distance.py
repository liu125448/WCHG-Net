# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =========================
# 配置区（按需改）
# =========================
MAT_PATH = "tsne_baseline.mat"
OUT_PATH = "scatter.svg"

# 横轴范围（和你原来一致）
X_MIN, X_MAX = 0.0, 1.5
NUM_POINTS = 1000

# 固定 y 轴范围与刻度（图二大概 0~7）
Y_LIM = (0.0, 4.0)
Y_TICKS = np.arange(0, 4.1, 1.0)

# KDE 平滑参数（越大越平滑；你也可以改成 'scott'/'silverman'）
KDE_BW = "scott"

# 是否画图二那种注释（D2、E2、虚线）
SHOW_ANNOTATION = True

# 颜色（尽量贴近图二）
COLOR_INTRA = "#e41a1c"  # 红
COLOR_INTER = "#377eb8"  # 蓝
ALPHA_FILL = 0.35

# =========================
# 工具函数
# =========================
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化，避免特征未归一导致 cosine 距离不准。"""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

# =========================
# 主流程
# =========================
def main():
    # 1) 读取 mat
    result = scipy.io.loadmat(MAT_PATH)
    query_feature = result["query_f"]
    query_label = result["query_label"].reshape(-1)
    gallery_feature = result["gallery_f"]
    gallery_label = result["gallery_label"].reshape(-1)

    # 2) 转成 float32 +（可选但推荐）归一化
    query_feature = np.asarray(query_feature, dtype=np.float32)
    gallery_feature = np.asarray(gallery_feature, dtype=np.float32)

    # 如果你确定特征已经归一化，可以注释掉下面两行
    query_feature = l2_normalize(query_feature)
    gallery_feature = l2_normalize(gallery_feature)

    # 3) 计算 cosine distance: 1 - cos_sim
    # cos_sim = gallery_feature @ query_feature.T
    cos_sim = np.matmul(gallery_feature, query_feature.T)  # (G, Q)
    distmat = 1.0 - cos_sim  # (G, Q)

    # 4) intra/inter mask（同类 vs 异类）
    # mask shape: (G, Q)
    mask = (gallery_label[:, None] == query_label[None, :])

    intra = distmat[mask].astype(np.float64)      # 同类距离
    inter = distmat[~mask].astype(np.float64)     # 异类距离

    # 5) KDE 曲线
    x = np.linspace(X_MIN, X_MAX, NUM_POINTS)

    kde_intra = gaussian_kde(intra, bw_method=KDE_BW)
    kde_inter = gaussian_kde(inter, bw_method=KDE_BW)

    y_intra = kde_intra(x)
    y_inter = kde_inter(x)

    # 6) 绘图（图二风格）
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)

    # 填充 + 轮廓线
    ax.fill_between(x, y_intra, 0, color=COLOR_INTRA, alpha=ALPHA_FILL, label="intra")
    ax.plot(x, y_intra, color=COLOR_INTRA, linewidth=1.4)

    ax.fill_between(x, y_inter, 0, color=COLOR_INTER, alpha=ALPHA_FILL, label="inter")
    ax.plot(x, y_inter, color=COLOR_INTER, linewidth=1.4)

    # 重叠区域（E2 的视觉效果：取 min 两条密度）
    overlap = np.minimum(y_intra, y_inter)
    ax.fill_between(x, overlap, 0, color="grey", alpha=0.18)

    # 7) 固定 y 轴范围与刻度（你要求的）
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)

    # x 轴范围也固定一下，方便对齐
    ax.set_xlim(X_MIN, X_MAX)

    # 不要坐标轴标题（你要求的）
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 图例（图二在左上角；你也可改 loc）
    ax.legend(loc="upper left", frameon=True)

    if SHOW_ANNOTATION:
        # 峰值位置（最大密度对应的 x）
        x_intra_peak = x[np.argmax(y_intra)]
        x_inter_peak = x[np.argmax(y_inter)]

        # 虚线
        ax.axvline(x_inter_peak, linestyle="--", linewidth=1.0, color="0.35")
        ax.axvline(x_intra_peak, linestyle="--", linewidth=1.0, color="0.35")

        # D2 箭头：两峰之间
        y_arrow = Y_LIM[0] + 0.75 * (Y_LIM[1] - Y_LIM[0])
        ax.annotate(
            "",
            xy=(x_intra_peak, y_arrow),
            xytext=(x_inter_peak, y_arrow),
            arrowprops=dict(arrowstyle="<->", linewidth=1.2, color="black"),
        )
        ax.text(
            (x_intra_peak + x_inter_peak) / 2.0,
            y_arrow + 0.15,
            r"$D_2$",
            ha="center",
            va="bottom",
        )

        # E2：在重叠区附近标一下（找 overlap 最大的位置）
        x_e2 = x[np.argmax(overlap)]
        y_e2 = overlap.max()
        ax.text(x_e2, min(y_e2 + 0.1, Y_LIM[1] - 0.2), r"$E_2$", ha="center", va="bottom")

    # 8) 输出
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_PATH, format="svg")
    plt.show()


if __name__ == "__main__":
    main()
