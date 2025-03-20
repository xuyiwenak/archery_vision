import os
import cv2
import logging
from datetime import datetime
import torch
from ultralytics import YOLO
import math
import csv
from src.enums.action_state import ActionState

class YoloBow:
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    angle_list = []
    release_angle = None

    @classmethod
    def process_video(cls, input_path, output_path):
        start_time = datetime.now()
        logger = logging.getLogger()

        # 创建CSV文件
        csv_path = output_path.rsplit('.', 1)[0] + '_data.csv'
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['时间(秒)', '帧号', '角度', '动作环节'])

        # 检查GPU是否可用
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ 使用设备: {device}")
        if device == 'cuda':
            logger.info(f"📊 GPU信息: {torch.cuda.get_device_name(0)}")

        logger.info(f"▶️ 开始处理 {input_path} → {output_path}")

        # 初始化模型并指定设备
        model_name = 'yolo11x-pose'
        model_path = f'data/models/{model_name}.pt'
        
        # 如果本地没有模型文件,则下载
        if not os.path.exists(model_path):
            logger.info(f"⏬ 下载 {model_name} 模型...")
            model = YOLO(f'{model_name}.pt')
        else:
            logger.info(f"📂 使用本地 {model_name} 模型")
            model = YOLO(model_path)
            
        model.to(device)
        logger.info(f"✅ 加载 {model_name} 模型到 {device} 设备")

        # 视频输入输出
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("❌ 无法打开视频文件")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(3)), int(cap.get(4)))

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        logger.info(f"📊 视频信息: {total_frames}帧 | {fps}FPS | 尺寸 {frame_size}")

        # 处理循环
        processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 计算当前视频时间
            current_time = processed / fps

            # 推理
            results = model.track(frame, imgsz=320, conf=0.5, verbose=False)[0]
            angle = 0
            action_state = ActionState.UNKNOWN
            # 获取关键点数据
            keypoints = results.keypoints
            if keypoints is not None:
                for person in keypoints.xy:
                    if len(person) < 1:
                        continue
                    # 关键点顺序：鼻子、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左脚踝、右脚踝
                    left_shoulder = person[5].cpu().numpy()
                    right_shoulder = person[6].cpu().numpy()
                    left_elbow = person[7].cpu().numpy()
                    right_elbow = person[8].cpu().numpy()
                    
                    # 绘制线段
                    cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_elbow[0]), int(left_elbow[1])), (0, 255, 0), 2)
                    cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_elbow[0]), int(right_elbow[1])), (0, 255, 0), 2)
                    # 计算夹角
                    angle = cls.calculate_angle(left_shoulder, left_elbow, right_shoulder, right_elbow)
                    # 获取动作环节
                    action_state = cls.judge_action(angle)
                    # 记录数据到CSV
                    csv_writer.writerow([f"{current_time:.2f}", processed, f"{angle:.2f}", action_state.value])
                    # 绘制角度值
                    cv2.putText(frame, f"Angle: {angle:.2f} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # 绘制技术环节
                    cv2.putText(frame, f"Technical process: {action_state.value} ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # 绘制帧序号
                    cv2.putText(frame, f"processed: {processed} ", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
        csv_file.close()

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ 处理完成: {processed}帧 | 总耗时 {total_time:.1f}s | "
            f"平均FPS {processed/total_time:.1f}\n"
            f"输出文件: {output_path}\n"
            f"数据文件: {csv_path}"
        )

    @staticmethod
    def calculate_angle(c, d, a, b):
        # 计算向量AB和CD
        vector_ab = (b[0] - a[0], b[1] - a[1])
        vector_cd = (d[0] - c[0], d[1] - c[1])
        
        # 计算点积和模长
        dot_product = vector_ab[0] * vector_cd[0] + vector_ab[1] * vector_cd[1]
        magnitude_ab = math.sqrt(vector_ab[0]**2 + vector_ab[1]**2)
        magnitude_cd = math.sqrt(vector_cd[0]**2 + vector_cd[1]**2)
        
        # 计算夹角（弧度）
        if magnitude_ab == 0 or magnitude_cd == 0:
            return 0
        cos_theta = dot_product / (magnitude_ab * magnitude_cd)
        # 防止由于浮点数精度问题导致cos_theta超出范围[-1, 1]
        cos_theta = max(min(cos_theta, 1), -1)
        angle_rad = math.acos(cos_theta)
        
        # 使用叉积判断角度方向
        cross_product = vector_ab[0] * vector_cd[1] - vector_ab[1] * vector_cd[0]
        
        # 转换为角度 (0-360范围)
        angle_deg = math.degrees(angle_rad)
        if cross_product < 0:
            angle_deg = 360 - angle_deg
            
        return angle_deg

    @classmethod
    def judge_action(cls, angle):
        """
        根据角度判断动作环节
        参数:
            angle (float): 计算出的角度值 (0-360范围)
        """
        cls.angle_list.append(angle)
        
        release_angle_threshold = 4.5  # 固势->撒放 角度骤增差值阈值

        if 330 <= angle < 360 or 0 < angle < 12:
            cls.release_angle = None  # 重置撒放角
            return ActionState.LIFT  # 举弓
        elif 12 <= angle < 150:
            return ActionState.DRAW  # 开弓
        elif cls.release_angle and cls.release_angle - release_angle_threshold <= angle <= 185:
            return ActionState.RELEASE  # 撒放
        elif 150 <= angle < 185:
            previous_angles = cls.angle_list[-4:-1]
            previous_angle = sum(previous_angles) / 3  # 取前三帧的平均值
            if min(previous_angles) >= 150 and 20 > angle - previous_angle >= release_angle_threshold:  # 固势下骤增角度可视为进入撒发环节 (撒放角)
                cls.release_angle = angle
                return ActionState.RELEASE  # 撒放
            return ActionState.SOLID  # 固势
        elif 185 <= angle < 215:
            return ActionState.RELEASE  # 撒放
        else:
            return ActionState.UNKNOWN