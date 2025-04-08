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