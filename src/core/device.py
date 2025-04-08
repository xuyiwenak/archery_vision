import torch
from src.core.log import logger

class Device:
    @classmethod
    def get_device(cls):
         # 自动选择最佳设备
        device = 'cuda' if torch.cuda.is_available() else \
                'mps' if torch.backends.mps.is_available() else 'cpu'
        if device == 'cuda':
            logger.info(f"🚀 使用CUDA加速: {torch.cuda.get_device_name(0)}")
        return device