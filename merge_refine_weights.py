import torch
from collections import OrderedDict

PATH_BACKBONE = "weights/Expv8_large_norefined_finetune_15.pt"  
PATH_NEW      = "weights/Expv8_large_20.pt"                     
PATH_OUT      = "weights/Expv8_large_finetune15_with_refiner20.pt"  

REFINER_PREFIX = "net.unet_refiner."  


def load_model_state(path):
    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt, ckpt["model_state"]
    else:
        return {}, ckpt


def main():
    ckpt_backbone, sd_backbone = load_model_state(PATH_BACKBONE)
    ckpt_new,      sd_new      = load_model_state(PATH_NEW)

    print(f"Backbone keys: {len(sd_backbone)}")
    print(f"New-arch keys: {len(sd_new)}")

    merged_state = OrderedDict(sd_backbone)

    refiner_keys = [k for k in sd_new.keys() if k.startswith(REFINER_PREFIX)]
    print(f"Found {len(refiner_keys)} refiner keys with prefix '{REFINER_PREFIX}'")

    if len(refiner_keys) == 0:
        print("没找到任何 net.unet_refiner.* 的权重，检查一下 PATH_NEW 是否正确。")
        return

    print("Example refiner keys:")
    for k in refiner_keys[:20]:
        print("  ", k)

    for k in refiner_keys:
        merged_state[k] = sd_new[k]

    new_ckpt = {}

    if isinstance(ckpt_new, dict):
        new_ckpt["epoch"]   = ckpt_new.get("epoch", None)
        new_ckpt["metrics"] = ckpt_new.get("metrics", None)
    else:
        new_ckpt["epoch"]   = None
        new_ckpt["metrics"] = None
    new_ckpt["model_state"] = merged_state

    torch.save(new_ckpt, PATH_OUT)
    print(f"Merged checkpoint saved to: {PATH_OUT}")
    print(f"Merged state_dict key count: {len(merged_state)}")


if __name__ == "__main__":
    main()
