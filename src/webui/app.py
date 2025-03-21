import gradio as gr
import os
from src.models.yolo_bow import YoloBow
import warnings
import signal
import sys

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
        # 处理视频
        YoloBow.process_video(video_path, output_path)
        return output_path
    except Exception as e:
        print(f"处理视频时发生错误: {str(e)}")
        return None            
    
def process_with_status(video):
    if not video:
        return None, "请先上传视频"
    result = process_video(video)
    if result:
        return result, "处理完成"
    else:
        return None, "处理失败，请检查控制台输出"

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

        process_btn.click(
            fn=process_with_status,
            inputs=[input_video],
            outputs=[output_video, status_text]
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