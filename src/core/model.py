import os

from ultralytics import YOLO

from src.core.log import logger

class Model: 
    @classmethod
    def get_model(cls):
        # 初始化模型
        model_name = 'yolo11x-pose'
        model_path = f'data/models/{model_name}.pt'
        # 如果本地没有模型文件,则下载
        if not os.path.exists(model_path):
            logger.info(f"⏬ 下载 {model_name} 模型...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model = YOLO(f'{model_name}.pt')
            model.export(format='pt', file=model_path)  # 保存模型到本地
        else:
            logger.info(f"📂 使用本地 {model_name} 模型")
            model = YOLO(model_path)
        return model