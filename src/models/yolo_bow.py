import cv2
import csv
from datetime import datetime

from src.core.device import Device
from src.core.model import Model
from src.core.pose import Pose
from src.enums.action_state import ActionState
from src.core.log import logger

class YoloBow:
    @classmethod
    def process_frames(cls, cap, model):
        # 定义帧缓冲区和批处理大小
        frame_buffer = []
        batch_size = 12  # 根据显存调整批处理大小
        while cap.isOpened():
            success, frame = cap.read()
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

        # 视频输入
        cap = cv2.VideoCapture(input_path)
        # 视频属性
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        logger.info(f"📊 视频信息: {total_frames}帧 | {fps}FPS | 尺寸 {frame_size}")
        # 视频输出
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)
        # 处理循环
        processed = 0
        
        csv_data = []
        for frame, result in cls.process_frames(cap, model):
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
                    # 计算夹角
                    angle = Pose.calculate_angle(left_shoulder, left_elbow, right_shoulder, right_elbow)
                    # 获取动作环节
                    action_state = Pose.judge_action(angle)
                    # 绘制角度值、技术环节、帧序号
                    cv2.putText(frame, f"Angle: {angle:.2f} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Technical process: {action_state.value} ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"processed: {processed} ", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    csv_data.append((processed, f"{angle:.2f}", action_state.value))

            writer.write(frame)

            # 进度日志
            processed += 1
            if processed % 30 == 0:  # 每30帧输出一次进度
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_log = processed / elapsed
                remain = (total_frames - processed) / fps_log if fps_log > 0 else 0
                logger.info(
                    f"⏳ 进度: {processed}/{total_frames} "
                    f"({processed/total_frames:.0%}) | "
                    f"耗时: {elapsed:.1f}s | "
                    f"剩余: {remain:.1f}s"
                )

        # 收尾工作
        cap.release()
        writer.release()

        # 创建CSV文件
        csv_path = output_path.rsplit('.', 1)[0] + '_data.csv'
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(('帧号', '角度', '动作环节'))
        csv_writer.writerows(csv_data)
        csv_file.close()

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ 处理完成: {processed}帧 | 总耗时 {total_time:.1f}s | "
            f"平均FPS {processed/total_time:.1f}\n"
            f"输出文件: {output_path}\n"
            f"数据文件: {csv_path}"
        )
