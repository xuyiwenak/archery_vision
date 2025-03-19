import os
import logging
from datetime import datetime
import cv2
import torch
from ultralytics import YOLO
import math


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

        # 检查GPU是否可用
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ 使用设备: {device}")
        if device == 'cuda':
            logger.info(f"📊 GPU信息: {torch.cuda.get_device_name(0)}")

        logger.info(f"▶️ 开始处理 {input_path} → {output_path}")

        # 初始化模型并指定设备
        model_name = 'yolo11x-pose'
        model = YOLO(f'{model_name}.pt')
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

            # # 将帧添加到批次中
            # frames_batch.append(frame)
            # # 当批次达到指定大小时，进行推理
            # if len(frames_batch) == batch_size:
            #     # 将批次中的帧转换为张量并送入模型
            #     frames_tensor = torch.from_numpy(np.array(frames_batch)).permute(0, 3, 1, 2).to(device).float() / 255.0
            #     results = model.track(frames_tensor, imgsz=320, conf=0.5, verbose=False)
            #     # 处理推理结果并写入视频
            #     for result in results:
            #         # 假设 result 是处理后的帧
            #         writer.write(result.cpu().numpy().astype(np.uint8))
            #     # 清空批次
            #     frames_batch = []

            # 推理
            results = model.track(frame, imgsz=320, conf=0.5, verbose=False)[0]
            angle = 0
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
                    # 绘制角度值
                    cv2.putText(frame, f"Angle: {angle:.2f} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # 绘制技术环节
                    cv2.putText(frame, f"Technical process: {cls.judge_action(angle)} ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            writer.write(frame)

            # writer.write(results.plot(boxes=False))

            # 进度日志
            processed += 1
            if processed % 30 == 0:  # 每30帧输出一次进度
                logger.info(angle)
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

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ 处理完成: {processed}帧 | 总耗时 {total_time:.1f}s | "
            f"平均FPS {processed/total_time:.1f}\n"
            f"输出文件: {output_path}"
        )

    @staticmethod
    def calculate_angleV1(a, b, c, d):
        # 计算向量AB和CD的夹角
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
        
        # 转换为角度
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    @staticmethod
    def calculate_angle(a, b, c, d):
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
        # 叉积的z分量（二维向量的叉积结果是一个标量）
        cross_product = vector_ab[0] * vector_cd[1] - vector_ab[1] * vector_cd[0]
        
        # 如果cross_product为正，角度为逆时针方向（正方向）
        # 如果为负，角度为顺时针方向（负方向）
        if cross_product < 0:
            angle_rad = -angle_rad
        
        # 转换为角度
        angle_deg = math.degrees(angle_rad)
        return - angle_deg

    @classmethod
    def judge_action(cls, angle):
        """
        根据角度判断动作环节
        参数:
            angle (float): 计算出的角度值（带符号）
        """
        cls.angle_list.append(angle)

        if -30 <= angle < 12:
            cls.release_angle = None # 重置撒放角
            return "Lift"  # 举弓
        elif 12 <= angle < 155:
            return "Draw"  # 开弓
        elif cls.release_angle and cls.release_angle <= angle < 180:
            return "Release"  # 撒放
        elif 155 <= angle < 180:
            previous_angle = cls.angle_list[-2]
            if previous_angle >= 150 and angle - previous_angle >= 4.5:  # 固势下骤增角度可视为进入撒发环节 (撒放角)
                cls.release_angle = angle
                return "Release"  # 撒放
            return "Solid"  # 固势
        elif -180 <= angle < -120:
            return "Release"  # 撒放
        else:
            return ''

def main():
    # 确保输入和输出目录存在
    input_dir = os.path.join('data', 'input')
    output_dir = os.path.join('data', 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 处理输入目录中的所有视频文件
    for filename in os.listdir(input_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f'output_{filename}')
            YoloBow.process_video(input_path, output_path)

if __name__ == "__main__":
    main()
