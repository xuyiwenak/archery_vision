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
        output_dir = os.path.join("data", "output")
        os.makedirs(output_dir, exist_ok=True)
        # 构建输出文件路径
        output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
        csv_path = os.path.join(output_dir, f"{base_name}_processed_data.csv")
        # 处理视频
        YoloBow.process_video(video_path, output_path)
        
        # 读取CSV数据
        angles = pd.read_csv(csv_path, encoding='utf8')
        return {
            "video_path": output_path,
            "angles": angles
        }
    except Exception as e:
        print(f"处理视频时发生错误: {str(e)}")
        raise e
        return None            
    

def process_with_status(video):
    if not video:
        return None, None, "请先上传视频", None
    result = process_video(video)
    if result:
        # 准备折线图数据(转换为DataFrame)
        slider = gr.Slider(minimum=0, maximum=len(result["angles"]), value=5, step=1, label="拖动滑块移动游标", interactive=True)
        return result["video_path"], result["angles"], "处理完成", slider
    else:
        return None, None, "处理失败，请检查控制台输出", None


# 视频播放时更新游标线
def update_cursor(video_state, data, slider):
    if data:
        return data, slider
    return data, slider


# todo 合并 process_video + process_with_status

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="射箭姿态分析", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎯 射箭姿态分析")
        # todo 模型选择
        # todo 选择 batch size
        # todo 左右手持弓选择，默认左手持弓
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="上传视频", sources="upload", interactive=True)
                process_btn = gr.Button("开始分析", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="分析结果", format="mp4", interactive=False)
            with gr.Column():            
                status_text = gr.Textbox(label="处理状态", interactive=False, value="等待上传视频...")
        
        with gr.Row():
            # slider = gr.Slider(0, 10, value=5, step=0.1, label="拖动滑块移动游标")
            slider = gr.Slider(minimum=0, maximum=100, value=5, step=1, label="拖动滑块移动游标", interactive=True)
        # 添加姿态角折线图
        with gr.Row():
            arm_plot = gr.LinePlot(label="双臂姿态角", x="帧号", y="角度", width=500, height=300)
            
        # slider.change(
        #     fn=update_cursor,
        #     inputs=[output_video, arm_plot, slider],
        #     outputs=[arm_plot, slider]
        # )

        process_btn.click(
            fn=process_with_status,
            inputs=[input_video],
            outputs=[output_video, arm_plot, status_text, slider]
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
