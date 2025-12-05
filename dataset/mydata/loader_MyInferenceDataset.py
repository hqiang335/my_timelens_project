from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tools.registery import DATASET_REGISTRY
from numba import jit

# 事件相机的目标分辨率
H_EVT, W_EVT = 612, 816

@jit(nopython=True)
def events_to_voxel_grid_xyttp(
    events_xyttp: np.ndarray,
    H: int = H_EVT,
    W: int = W_EVT,
    bins: int = 16,
) -> np.ndarray:
    """
    将单个帧间隔内的 [x,y,t,p] 事件列表转换为体素网格 [bins, H, W]
    - events_xyttp: 形状为 (N, 4) 的 numpy 数组，包含 [x, y, t, p]，其中 p ∈ {-1, 0, 1}
    - bins: 该时间间隔内的总时间桶数
    返回形状为 [bins, H, W] 的 float32 体素网格
    """
    voxel = np.zeros((bins, H, W), dtype=np.float32)
    if events_xyttp.size == 0:
        return voxel
    
    x = events_xyttp[:, 0].astype(np.float32)
    y = events_xyttp[:, 1].astype(np.float32)
    t = events_xyttp[:, 2].astype(np.float32)
    p = events_xyttp[:, 3].astype(np.float32)

    # 坐标边界检查（安全性）
    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not np.all(m):
        x, y, t, p = x[m], y[m], t[m], p[m]

    if x.size == 0:
        return voxel

    # 计算时间步长（与 sample_events_to_grid 一致）
    t_start = t.min()
    t_end = t.max()
    t_step = (t_end - t_start + 1) / bins
    
    # 处理每个事件（与 sample_events_to_grid 一致）
    for i in range(len(x)):
        d_x, d_y, d_t, d_p = x[i], y[i], t[i], p[i]
        d_x_low, d_y_low = int(d_x), int(d_y)
        d_t_adjusted = d_t - t_start
        ind = int(d_t_adjusted // t_step)
        
        # 确保索引在有效范围内
        if ind < 0 or ind >= bins:
            continue
            
        x_weight = d_x - d_x_low
        y_weight = d_y - d_y_low
        pv = d_p  # 使用原始极性值（与 sample_events_to_grid 一致）
        
        # 双线性插值累加（与 sample_events_to_grid 一致）
        if d_y_low < H and d_x_low < W:
            voxel[ind, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * pv
        if d_y_low + 1 < H and d_x_low < W:
            voxel[ind, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * pv
        if d_x_low + 1 < W and d_y_low < H:
            voxel[ind, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * pv
        if d_y_low + 1 < H and d_x_low + 1 < W:
            voxel[ind, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * pv
            
    return voxel
# def events_to_voxel_grid_xyttp(
#     events_xyttp: np.ndarray,
#     H: int = H_EVT,
#     W: int = W_EVT,
#     bins: int = 128,
# ) -> np.ndarray:
#     """
#     将单个帧间隔内的 [x,y,t,p] 事件列表转换为体素网格 [bins, H, W]
#     - events_xyttp: 形状为 (N, 4) 的 numpy 数组，包含 [x, y, t, p]，其中 p ∈ {-1, 0, 1}
#     - bins: 该时间间隔内的总时间桶数
#     返回形状为 [bins, H, W] 的 float32 体素网格
#     """
#     voxel = np.zeros((bins, H, W), dtype=np.float32)
#     if events_xyttp.size == 0:
#         return voxel
    
#     x = events_xyttp[:, 0].astype(np.int32)
#     y = events_xyttp[:, 1].astype(np.int32)
#     t = events_xyttp[:, 2].astype(np.int64)
#     p = events_xyttp[:, 3].astype(np.int8)

#     # 坐标边界检查（安全性）
#     m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
#     if not np.all(m):
#         x, y, t, p = x[m], y[m], t[m], p[m]

#     if x.size == 0:
#         return voxel

#     # 将时间归一化到 [0, 1] 区间；单时间戳特殊情况 -> 全部放入第0个桶
#     t_min = t.min()
#     t_max = t.max()
#     if t_max == t_min:
#         bin_idx = np.zeros_like(t, dtype=np.int32)
#     else:
#         t_norm = (t - t_min) / float(t_max - t_min)
#         bin_idx = np.floor(t_norm * (bins - 1e-7)).astype(np.int32)
#         np.clip(bin_idx, 0, bins - 1, out=bin_idx)

#     # 累加计数（支持正负极性）
#     np.add.at(voxel, (bin_idx, y, x), p.astype(np.float32))
#     return voxel

def build_loader_infer(
        aps_dir: str = "mydata/test/scene1/aps_png",
        ev_dir:  str = "mydata/test/scene1/events",
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs
    ):
        ds = loader_MyInferenceDataset(aps_dir=aps_dir, ev_dir=ev_dir, **kwargs)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)

@DATASET_REGISTRY.register()
class loader_MyInferenceDataset(Dataset):
    def __init__(self, params, training: bool = False,):
        # 从 params 中获取路径配置
        self.aps_dir = Path(params.paths.test_rgb)  # 使用 params 中的路径
        self.ev_dir = Path(params.paths.test_evs)   # 使用 params 中的路径
        
        # 从 params 中获取其他配置
        self.interp_ratio = int(params.interp_ratio)
        self.events_bins = int(params.events_channel)
        self.scene = params.scene_name if hasattr(params, 'scene_name') else self.aps_dir.parent.name
        
        # 加载APS帧文件
        self.aps_files = sorted(self.aps_dir.glob("*.png"))
        assert len(self.aps_files) >= 2, "至少需要2个APS帧"

        # 构建帧名 -> 事件文件路径的映射
        self.ev_map: Dict[str, Path] = {}
        for p in sorted(self.ev_dir.glob("*.npy")):
            stem = p.stem
            self.ev_map[stem] = p

        # 构建有效索引
        self.indices: List[int] = []
        for i in range(len(self.aps_files) - 1):
            s = self.aps_files[i].stem
            if s in self.ev_map:
                self.indices.append(i)
        # assert len(self.indices) > 0, "未找到匹配的（帧i, 帧i+1, 事件i）组合"

    def __len__(self) -> int:
        return len(self.indices)

    def _load_img_resized(self, path: Path) -> np.ndarray:
    
        img = Image.open(path).convert("RGB")
        img = img.resize((W_EVT, H_EVT), resample=Image.BILINEAR)
        return np.array(img)  # H,W,3

    def __getitem__(self, k: int) -> Dict[str, Any]:
        i = self.indices[k]
        f0, f1 = self.aps_files[i], self.aps_files[i+1]
        stem0, stem1 = f0.stem, f1.stem

        # 加载并下采样图像
        im0_np = self._load_img_resized(f0)  # H,W,3
        im1_np = self._load_img_resized(f1)

        # 转换为 [0,1] 范围的浮点张量，并调整为 CHW 格式
        im0 = torch.from_numpy(im0_np.astype(np.float32) / 255.0).permute(2,0,1).contiguous()
        im1 = torch.from_numpy(im1_np.astype(np.float32) / 255.0).permute(2,0,1).contiguous()

        # 加载事件数据并转换为体素网格
        ev_path = self.ev_map[stem0]
        evs = np.load(ev_path, allow_pickle=False)  # (N,4) int64
        voxel = events_to_voxel_grid_xyttp(evs, H=H_EVT, W=W_EVT, bins=self.events_bins)
        events = torch.from_numpy(voxel)  # [Ce,H,W], float32

        sample = {
            "im0": im0,                # [C,H,W]
            "im1": im1,                # [C,H,W]
            "events": events,          # [Ce,H,W]
            "gts": torch.empty(0),     # 推理时无真值
            "rgb_name": [f0.name, f1.name],  # 改为列表的列表
            "folder": self.scene,    # 改为单元素列表
            "interp_ratio": self.interp_ratio,  # 添加插值比例
        }
        return sample