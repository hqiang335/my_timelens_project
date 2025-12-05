import torch
import torch.nn as nn

from tools.registery import PARAM_REGISTRY
from tools.model_deparse import deparse_model


def check_one(param_name, model_name, ckpt_path):
    # 伪造一个和 run_network 差不多的 args
    class Args:
        pass

    args = Args()
    args.param_name = param_name
    args.model_name = model_name
    args.model_pretrained = ckpt_path
    args.extension = ""   # 无所谓

    # 从 PARAM_REGISTRY 里拿到 param 函数，然后构造 params
    param_fn = PARAM_REGISTRY.get(param_name)
    params = param_fn(args)
    params.model_config.model_pretrained = ckpt_path  # 强制用我们指定的权重

    # 用原来的 deparse_model 构造模型（保证和训练时一模一样）
    model, epoch, _ = deparse_model(params)
    model = model.cpu()

    print("\n==============================")
    print(f"Checkpoint: {ckpt_path}")
    print(f"param_name: {param_name}, model_name: {model_name}")
    print(f"Loaded epoch in ckpt: {epoch}")
    print("------------------------------")

    # 1) 直接看 config 里的 echannel
    echannel = getattr(params.model_config.define_model, "echannel", None)
    print(f"echannel in config = {echannel}")

    # 2) 看模型里第一层 Conv2d 的 in_channels
    first_conv = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            first_conv = (name, m.in_channels, m.out_channels, m.kernel_size)
            break
    if first_conv:
        name, in_c, out_c, k = first_conv
        print(f"first Conv2d: {name}, in={in_c}, out={out_c}, k={k}")

    # 3) 查一下是否有 unet_refiner
    has_refiner = any("unet_refiner" in name for name, _ in model.named_modules())
    print(f"has 'unet_refiner' ? -> {has_refiner}")


if __name__ == "__main__":
    # ① 检查你刚训练的 x4_e16 的第 0 个 checkpoint
    check_one(
        param_name="params_x4_e16",          # 你新建的 param 函数名
        model_name="Expv8_large",
        ckpt_path="RGB_resOut_HQEVFI/Expv8_largex3_unetRefiner/weights/Expv8_large_0.pt",
    )

    # ② 检查原始 HQEVFI 预训练权重（128 通道、无 refiner 的那版）
    check_one(
        param_name="traintest_RC_smallmix_lpips",  # 原来跑 HQEVFI 的 param 函数名
        model_name="Expv8_large",
        ckpt_path="weights/Expv8_large_HQEVFI.pt",
    )
