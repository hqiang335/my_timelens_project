import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# ---------- 配置：根据你之前的设计 ----------
# 假设输入是 [I_hat, I0, I1] 拼在一起：256x256x9
layers = [
    {"name": "Input\n[Ĩτ, I0, I1]", "shape": "256×256×9"},
    {"name": "Enc1", "shape": "256×256×32"},
    {"name": "Down1\n+ Enc2", "shape": "128×128×64"},
    {"name": "Bottleneck", "shape": "64×64×128"},
    {"name": "Up2\n+ Dec2", "shape": "128×128×64"},
    {"name": "Up1\n+ Dec1", "shape": "256×256×32"},
    {"name": "Output\nResidual / RGB", "shape": "256×256×3"},
]

# 每个 box 的宽度和高度（图像坐标系单位）
box_w = 1.5
box_h = 1.0
x_spacing = 0.7  # box 左右间距
y_center = 0     # 所有 box 放在同一条水平线上

# ---------- 画图 ----------
fig, ax = plt.subplots(figsize=(10, 3))

# 记录每个 box 的中心位置，方便画箭头 / skip connection
centers = []

for i, layer in enumerate(layers):
    x = i * (box_w + x_spacing)
    y = y_center - box_h / 2

    # 画矩形
    rect = Rectangle(
        (x, y), box_w, box_h,
        linewidth=1.5,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(rect)

    # 文字：层名放在上半部分，shape 放在下半部分
    ax.text(
        x + box_w / 2, y + box_h * 0.65,
        layer["name"],
        ha="center", va="center", fontsize=8
    )
    ax.text(
        x + box_w / 2, y + box_h * 0.30,
        layer["shape"],
        ha="center", va="center", fontsize=8, color="dimgray"
    )

    centers.append((x + box_w, y_center))  # 右侧中心点

# ---------- 箭头（顺序连接） ----------
for i in range(len(centers) - 1):
    x0, y0 = centers[i]
    x1, y1 = centers[i + 1]
    ax.add_patch(
        FancyArrow(
            x0 + 0.05, y0,
            (x1 - box_w - x_spacing/2) - (x0 + 0.05),
            0,
            width=0.01,
            length_includes_head=True,
            head_width=0.12,
            head_length=0.15,
            color="black"
        )
    )

# ---------- 可选：画两条 skip connection（UNet 味道） ----------
# Enc1 -> Up1+Dec1
x_enc1 = (0 * (box_w + x_spacing)) + box_w
x_up1  = (5 * (box_w + x_spacing))
y_skip = y_center + 0.9  # 在上面画一条弧形线

ax.annotate(
    "", xy=(x_up1, y_skip), xytext=(x_enc1, y_skip),
    arrowprops=dict(arrowstyle="->", linestyle="--", color="gray")
)
ax.text((x_enc1 + x_up1)/2, y_skip + 0.1, "skip", ha="center", va="bottom", fontsize=7, color="gray")

# Down1+Enc2 -> Up2+Dec2
x_enc2 = (2 * (box_w + x_spacing)) + box_w
x_up2  = (4 * (box_w + x_spacing))
y_skip2 = y_center + 0.5

ax.annotate(
    "", xy=(x_up2, y_skip2), xytext=(x_enc2, y_skip2),
    arrowprops=dict(arrowstyle="->", linestyle="--", color="gray")
)

# ---------- 画布设置 ----------
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
plt.savefig("unet_refiner_arch.png", dpi=300, bbox_inches="tight")
plt.show()
