import os
from src.models.yolo_bow import YoloBow
from src.core.log import logger
def main():
    # 确保输入和输出目录存在
    input_dir = os.path.join('data', 'input')
    output_dir = os.path.join('data', 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    default_model_name = 'yolo11x-pose'
    # 处理输入目录中的所有视频文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f'output_{filename}')
            YoloBow.process_video(input_path, output_path, default_model_name)

if __name__ == "__main__":
    main()
