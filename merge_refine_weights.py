import torch
from collections import OrderedDict

# ========= 根据你的实际路径改这三个 =========
PATH_BACKBONE = "weights/Expv8_large_norefined_finetune_15.pt"  # 作为主干的 ckpt
PATH_NEW      = "weights/Expv8_large_20.pt"                     # 含 unet_refiner 的 ckpt
PATH_OUT      = "weights/Expv8_large_finetune15_with_refiner20.pt"  # 合并后的输出

REFINER_PREFIX = "net.unet_refiner."  # 你的轻量 UNet 模块统一前缀


def load_model_state(path):
    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt, ckpt["model_state"]
    else:
        # 兼容直接保存 state_dict 的情况
        return {}, ckpt


def main():
    # 1. 加载两个 checkpoint
    ckpt_backbone, sd_backbone = load_model_state(PATH_BACKBONE)
    ckpt_new,      sd_new      = load_model_state(PATH_NEW)

    print(f"Backbone keys: {len(sd_backbone)}")
    print(f"New-arch keys: {len(sd_new)}")

    # 2. 以 finetune_15 的 state_dict 为基底
    merged_state = OrderedDict(sd_backbone)

    # 3. 从 new-arch 里拿出所有 net.unet_refiner.* 的参数
    refiner_keys = [k for k in sd_new.keys() if k.startswith(REFINER_PREFIX)]
    print(f"Found {len(refiner_keys)} refiner keys with prefix '{REFINER_PREFIX}'")

    if len(refiner_keys) == 0:
        print("⚠️ 没找到任何 net.unet_refiner.* 的权重，检查一下 PATH_NEW 是否正确。")
        return

    # 方便你 sanity check 一下
    print("Example refiner keys:")
    for k in refiner_keys[:20]:
        print("  ", k)

    # 4. 覆盖/添加到 merged_state 里
    for k in refiner_keys:
        merged_state[k] = sd_new[k]

    # 5. 重新组织成新的 ckpt 结构
    new_ckpt = {}

    # epoch / metrics 用哪个都行，看你习惯：
    # 方案 A：用 refiner 训练完 20 的信息
    if isinstance(ckpt_new, dict):
        new_ckpt["epoch"]   = ckpt_new.get("epoch", None)
        new_ckpt["metrics"] = ckpt_new.get("metrics", None)
    else:
        new_ckpt["epoch"]   = None
        new_ckpt["metrics"] = None

    # 方案 B（可选）：如果你更想标记为“15+refiner”，也可以手动写：
    # new_ckpt["epoch"] = 20   # 或者 "15+refiner"
    # new_ckpt["metrics"] = None

    new_ckpt["model_state"] = merged_state

    # 6. 保存
    torch.save(new_ckpt, PATH_OUT)
    print(f"✅ Merged checkpoint saved to: {PATH_OUT}")
    print(f"Merged state_dict key count: {len(merged_state)}")

    # 7. 可选：加载一下测试 strict=False，看一下 missing/unexpected keys
    print("\nOptional: 你可以在模型代码里这样加载检查：")
    print(f"""
    ckpt = torch.load("{PATH_OUT}", map_location="cpu")
    model = build_your_model(...)  # 新架构（包含 net.unet_refiner）
    msg = model.load_state_dict(ckpt["model_state"], strict=False)
    print(msg)
    """)


if __name__ == "__main__":
    main()
