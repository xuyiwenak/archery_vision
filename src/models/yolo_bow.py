import cv2
import csv
from datetime import datetime
import pandas as pd

from src.core.device import Device
from src.core.model import Model
from src.core.pose import Pose
from src.core.video import Video
from src.enums.action_state import ActionState
from src.core.log import logger, log_process

class YoloBow:
    @classmethod
    @log_process
    def process_frames(cls, video, model):
        # 定义帧缓冲区和批处理大小
        frame_buffer = []
        batch_size = 12  # 根据显存调整批处理大小
        while video.capture.isOpened():
            success, frame = video.capture.read()
            if not success: 
                if frame_buffer:
                    results = model.track(frame_buffer, imgsz=320, conf=0.5, verbose=False, stream=True)
                    for k, result in enumerate(results):
                        yield frame_buffer[k], result
                break
            # 将帧添加到缓冲区
            frame_buffer.append(frame)
            # 当缓冲区达到批处理大小时，进行批量处理
            if len(frame_buffer) == batch_size:
                # 批量处理帧
                results = model.track(frame_buffer, imgsz=320, conf=0.5, verbose=False, stream=True)
                # 处理结果（例如绘制轨迹等）
                for k, result in enumerate(results):
                    yield frame_buffer[k], result
                frame_buffer = []

    @classmethod
    def process_video(cls, input_path, output_path):
        start_time = datetime.now()
        logger.info(f"▶️ 开始处理 {input_path} → {output_path}")

        device = Device.get_device()
        model = Model.get_model()
        model.to(device)
        logger.info(f"✅ 加载 {model.model_name} 模型到 {device} 设备")

        video = Video(input_path, output_path)
        # 数据记录 角度值、技术环节、帧序号
        records = pd.DataFrame(columns=['帧号', '角度', '动作环节'])
        # 处理循环
        processed = 0
        for frame, result in cls.process_frames(video, model):
            frame = result.plot(boxes=False)
            angle = 0
            action_state = ActionState.UNKNOWN
            # 获取关键点数据
            keypoints = result.keypoints
            if keypoints is not None:
                for person in keypoints.xy:
                    if len(person) < 1:
                        continue
                    # 关键点顺序：鼻子、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左脚踝、右脚踝
                    left_shoulder = person[5].cpu().numpy()
                    right_shoulder = person[6].cpu().numpy()
                    left_elbow = person[7].cpu().numpy()
                    right_elbow = person[8].cpu().numpy()
                    # todo 未完整识别到两臂坐标时不继续做分析处理，跳过进入下一帧
                    
                    # # 绘制线段 todo 可选是否绘制双臂
                    # cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_elbow[0]), int(left_elbow[1])), (0, 255, 0), 2)
                    # cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_elbow[0]), int(right_elbow[1])), (0, 255, 0), 2)
                    
                    angle = Pose.calculate_angle(left_shoulder, left_elbow, right_shoulder, right_elbow)  # 计算夹角
                    action_state = Pose.judge_action(angle)  # 获取动作环节
                    # 绘制角度值、技术环节、帧序号
                    frame = cls.put_texts(frame, (f"processed: {processed}", f"Angle: {angle:.2f} deg", f"Technical process: {action_state.value}"))   
                    # 记录数据                    
                    pd.concat((records, pd.DataFrame((processed, f"{angle:.2f}", action_state.value))))

            video.writer.write(frame)

        # 收尾工作
        video.close()
        # 创建CSV文件
        csv_path = output_path.rsplit('.', 1)[0] + '_data.csv'
        records.to_csv(csv_path, index=False, encoding='utf-8')

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ 处理完成: {processed}帧 | 总耗时 {total_time:.1f}s | "
            f"平均FPS {processed/total_time:.1f}\n"
            f"输出文件: {output_path}\n"
            f"数据文件: {csv_path}"
        )

    @classmethod
    def put_texts(cls, frame, texts):
        for k, text in enumerate(texts):
            cv2.putText(frame, text, (50, (k + 1) * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame