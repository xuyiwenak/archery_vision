import torch
from src.core.log import logger

class Device:
    @classmethod
    def get_device(cls, device_name='auto'):
        logger.info(f"🚀 使用MAC mps加速: {torch.mps.is_available()}")
        if device_name == 'auto':
            # 自动选择最佳设备
            if torch.backends.mps.is_available():
                device = 'mps'
                logger.info(f"🚀 使用MPS加速: {torch.cuda.get_device_name(0)}")
            elif torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"🚀 使用CUDA加速: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.info("🚀 使用CPU")
        else:
            device = device_name
        return device