from models.BaseModel import  BaseModel
from .archs import define_network
import numpy as np
import torch
from tools.registery import MODEL_REGISTRY
import os
from torch.nn import functional as F


@MODEL_REGISTRY.register()
class Expv8_large_infer(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.net = define_network(params.model_config.define_model)
        # self.net.debug = self.debug
        # self.grad_cache = {}
        # self.real_interp = None if 'real_interp' not in params.keys() else params.real_interp

        # === 添加推理专用属性 ===
        self.interp_ratio = params.interp_ratio  # 保存插值比例
        
    def metrics_init(self):
        """
        覆盖 BaseModel.metrics_init，推理模式下不构建任何度量/损失。
        这样就不会访问 params.validation_config.losses。
        """
        self.train_metrics = {}
        self.val_metrics = {}

    def net_validation(self, data_in, epoch):
        self.eval() # 设置为评估模式
        with torch.no_grad(): # 禁用梯度计算
            # 1. 数据提取
            left_frame, right_frame, events = data_in['im0'].cuda(), \
                data_in['im1'].cuda(), data_in['events'].cuda()
            
            # 获取插值比例
            interp_ratio = data_in['interp_ratio'][0].item()  # 取第一个样本的插值比例
            
            # 生成所有中间帧的时间点 (1到15)
            end_tlist = list(range(1, interp_ratio))
            
            # 进行推理，生成所有中间帧
            # 模型返回一个元组，我们取第一个元素（插值帧张量）
            recon_tuple = self.forward(left_frame, right_frame, events, interp_ratio, end_tlist)
            recon = recon_tuple[0]  # 提取实际的插值帧张量
            
            # 保存结果
            self.save_inference_results(recon, data_in, epoch)
        return

    def save_inference_results(self, res, data_in, epoch):
        """保存推理结果到文件"""
        import os
        import torchvision
        
        # 创建按场景组织的目录
        scene = os.path.basename(data_in['folder'][0])
        save_dir = os.path.join(self.val_im_path, f"scene_{scene}", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取帧信息
        frame0_name = os.path.splitext(data_in['rgb_name'][0][0])[0]
        frame1_name = os.path.splitext(data_in['rgb_name'][-1][0])[0]
        
        # 保存所有插值帧
        for n in range(res.shape[1]):
            # 生成基于时间戳的文件名
            timestamp = n / res.shape[1]  # 0到1之间的时间位置
            save_name = f"{frame0_name}_to_{frame1_name}_t{timestamp:.4f}.png"
            
            # 保存结果图像
            torchvision.utils.save_image(
                res[0, n].detach().cpu().clamp(0, 1),
                os.path.join(save_dir, save_name)
            )

    def forward(self, left_frame, right_frame, events, interp_ratio, end_tlist=None):
        if end_tlist is None:
            # 默认只生成中间帧
            end_tlist = range(interp_ratio - 1, interp_ratio)
        return self.net(torch.cat((left_frame, right_frame), 1), events, interp_ratio, end_tlist)
