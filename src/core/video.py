import os
import cv2

from src.core.log import logger


class Video:
    def __init__(self, input_path, output_path) -> None:
        self.input_path = input_path
        self.output_path = output_path
         # 视频输入
        self.capture = cv2.VideoCapture(input_path)
        # 视频属性
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_size = (int(self.capture.get(3)), int(self.capture.get(4)))
        logger.info(f"📊 视频信息: {self.total_frames}帧 | {self.fps}FPS | 尺寸 {self.frame_size}")
        # 视频输出
        self.writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), self.fps, self.frame_size)
        self.processed = 0

    def close(self):
        self.capture.release()
        self.writer.release()

    @staticmethod
    def extract_frame(video_path, frame_number):
        """从视频中提取指定帧号的图像"""
        if not video_path or not os.path.exists(video_path):
            return None

        try:
            cap = cv2.VideoCapture(video_path)
            # 设置帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            # 读取指定帧
            success, frame = cap.read()
            cap.release()
            
            if success:
                # 将BGR格式转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                print(f"无法读取帧 {frame_number}")
                return None
        except Exception as e:
            print(f"提取帧时发生错误: {str(e)}")
            return None