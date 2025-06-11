import cv2
from datetime import datetime
import pandas as pd
import numpy as np

from src.core.device import Device
from src.core.model import Model
from src.core.pose import Pose
from src.core.video import Video
from src.enums.action_state import ActionState
from src.core.log import logger, log_process

class YoloBow:
    @classmethod
    @log_process
    def process_frames(cls, video, model, batch_size):
        # 定义帧缓冲区和批处理大小
        frame_buffer = []
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
    def process_video(cls, input_path, output_path, model_name='yolo11x-pose', device_name='auto', batch_size=12):
        start_time = datetime.now()
        logger.info(f"▶️ 开始处理 {input_path} → {output_path}")

        device = Device.get_device(device_name)
        model = Model.get_model(model_name)
        model.to(device)
        logger.info(f"✅ 加载 {model.model_name} 模型到 {device} 设备")

        video = Video(input_path, output_path)
        # 数据记录 双臂姿态角、脊柱倾角、技术环节、帧序号
        records = pd.DataFrame(columns=['帧号', '双臂姿态角', '脊柱倾角', '动作环节'])
        # 处理循环
        for processed, (frame, result) in enumerate(cls.process_frames(video, model, batch_size)):
            frame = result.plot(boxes=False)
            arm_angle = 0
            spine_angle = 0
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
                    left_hip = person[11].cpu().numpy()
                    right_hip = person[12].cpu().numpy()

                    # 计算肩部和髋部的中点
                    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
                    hip_midpoint = (left_hip + right_hip) / 2
                    
                    # 计算脊柱向量与垂直线的夹角
                    spine_vector = shoulder_midpoint - hip_midpoint
                    vertical_vector = np.array([0, -1])  # 垂直向上的单位向量
                    spine_angle = Pose.calculate_angle(hip_midpoint, shoulder_midpoint, hip_midpoint, hip_midpoint + vertical_vector)
                    if spine_angle > 180:  # 将角度转换到 -180 到 180 度范围
                        spine_angle = spine_angle - 360

                    # todo 未完整识别到两臂坐标时不继续做分析处理，跳过进入下一帧
                    # # 绘制线段 todo 可选是否绘制双臂
                    # cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_elbow[0]), int(left_elbow[1])), (0, 255, 0), 2)
                    # cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_elbow[0]), int(right_elbow[1])), (0, 255, 0), 2)
                    # 绘制脊柱线段
                    cv2.line(frame, (int(hip_midpoint[0]), int(hip_midpoint[1])), 
                            (int(shoulder_midpoint[0]), int(shoulder_midpoint[1])), (255, 0, 0), 2)

                    arm_angle = Pose.calculate_angle(left_shoulder, left_elbow, right_shoulder, right_elbow)  # 计算双臂夹角
                    action_state = Pose.judge_action(arm_angle)  # 获取动作环节
                    # 绘制角度值、技术环节、帧序号
                    frame = cls.put_texts(frame, (
                        f"processed: {processed}", 
                        f"Arm Angle: {arm_angle:.2f} deg",
                        f"Spine Tilt: {spine_angle:.2f} deg", 
                        f"Technical process: {action_state.value}"
                    ))   
                    # 记录数据                   
                    records.loc[len(records)] = [processed, round(arm_angle, 2), round(spine_angle, 2), action_state.value]

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