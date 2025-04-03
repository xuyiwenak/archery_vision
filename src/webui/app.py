import gradio as gr
import os
import pandas as pd
from src.models.yolo_bow import YoloBow
import warnings
import signal
import sys
import csv

def process_video(video_path):
    """处理上传的视频文件"""
    if not video_path:
        return None
    
    try:
        # 获取输入视频的文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # 创建输出目录（如果不存在）
        output_dir = "output"  # fixme 改为 data/output
        os.makedirs(output_dir, exist_ok=True)
        # 构建输出文件路径
        output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
        csv_path = os.path.join(output_dir, f"{base_name}_processed_data.csv")
        # 处理视频
        YoloBow.process_video(video_path, output_path)
        
        # 读取CSV数据
        angles = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                angles.append({
                    "frame": int(row['帧号']),
                    "angle": float(row['角度']),
                    "state": row['动作环节']
                })
                
        return {
            "video_path": output_path,
            "angles": angles
        }
    except Exception as e:
        print(f"处理视频时发生错误: {str(e)}")
        return None            
    
def process_with_status(video):
    if not video:
        return None, None, None, "请先上传视频"
    result = process_video(video)
    if result:
        # 准备折线图数据(转换为DataFrame)
        left_arm_df = pd.DataFrame([{"frame": a["frame"], "angle": a["angle"]} for a in result["angles"]])
        right_arm_df = pd.DataFrame([{"frame": a["frame"], "angle": 360 - a["angle"]} for a in result["angles"]])
        return result["video_path"], left_arm_df, right_arm_df, "处理完成"
    else:
        return None, None, None, "处理失败，请检查控制台输出"

# todo 合并 process_video + process_with_status

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="射箭姿态分析", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎯 射箭姿态分析")
        # todo 模型选择
        # todo 设备选择，默认cpu
        # todo 选择 batch size
        # todo 左右手持弓选择，默认左手持弓
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="上传视频", sources="upload", interactive=True)
                process_btn = gr.Button("开始分析", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="分析结果", format="mp4", interactive=False)
                
        # 添加处理状态显示
        with gr.Row():
            status_text = gr.Textbox(label="处理状态", interactive=False, value="等待上传视频...")
        
        # 添加进度条
        progress_bar = gr.Slider(label="处理进度", interactive=False, minimum=0, maximum=100, step=1)
        
        # 添加姿态角折线图
        with gr.Row():
            left_arm_plot = gr.LinePlot(label="左臂姿态角", x="frame", y="angle", width=500, height=300)
            right_arm_plot = gr.LinePlot(label="右臂姿态角", x="frame", y="angle", width=500, height=300)
            
        # 视频播放时更新游标线
        def update_cursor(video_state):
            try:
                if isinstance(video_state, dict) and video_state.get("playing"):
                    current_time = video_state.get("time", 0)
                    duration = video_state.get("duration", 1)
                    progress = (current_time / duration) * 100 if duration > 0 else 0
                    return {"x": [progress], "y": [progress]}, {"x": [progress], "y": [progress]}
            except Exception as e:
                print(f"更新游标时出错: {e}")
            return {"x": [0], "y": [0]}, {"x": [0], "y": [0]}
            
        # 添加示例折线图
        with gr.Row():
            example_df = pd.DataFrame({
                "frame": range(1, 101),
                "angle": range(0, 100)
            })
            example_plot = gr.LinePlot(
                label="示例折线图", 
                x="frame", 
                y="angle", 
                width=500, 
                height=300,
                value=example_df
            )
            
        output_video.change(
            fn=update_cursor,
            inputs=[output_video],
            outputs=[left_arm_plot, right_arm_plot]
        )

        process_btn.click(
            fn=process_with_status,
            inputs=[input_video],
            outputs=[output_video, left_arm_plot, right_arm_plot, status_text]
        )
        
    return app

if __name__ == "__main__":
    # 创建并启动UI
    app = create_ui()
    app.queue()  # 启用队列处理以提高稳定性
    app.launch(
        server_name="127.0.0.1",  # 只监听本地连接
        show_error=True,
        quiet=True,  # 减少控制台输出
        share=False
    )
