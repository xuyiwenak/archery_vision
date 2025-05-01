import gradio as gr
import os
import pandas as pd
import cv2
from src.core.video import Video
from src.models.yolo_bow import YoloBow

def process_video(video_path, user_options):
    """处理上传的视频文件"""
    if not video_path:
        return "请先上传视频", *[None]*7

    # 获取输入视频的文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 创建输出目录（如果不存在）
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    # 构建输出文件路径
    output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
    csv_path = os.path.join(output_dir, f"{base_name}_processed_data.csv")
    # 处理视频
    YoloBow.process_video(video_path, output_path, 
                            model_name=user_options['model_dropdown'], device_name=user_options['device_dropdown'],
                            batch_size=user_options['batch_size'])
    # 读取CSV数据
    angles = pd.read_csv(csv_path, encoding='utf8')
    angles['定位'] = ''
    angles['角速度'] = angles['角度'].diff()  # 计算角速度 (度/帧)
    angles['角加速度'] = angles['角速度'].diff()  # 计算角加速度 (度/帧^2)
    # 准备折线图数据(转换为DataFrame)
    slider = gr.Slider(minimum=0, maximum=len(angles), value=5, step=1, label="拖动滑块移动游标", interactive=True)
    # 提取第一帧作为初始帧
    initial_frame = Video.extract_frame(output_path, 5)
    return "处理完成", output_path,  slider, initial_frame, *[angles]*4

    

# 视频播放时更新游标线
def update_cursor(data, slider):
    if data:
        # data['data'] = [point for point in data['data'] if point[3] != '游标']  # 删除之前的游标数据
        # data['data'].extend([[slider, 0, '', '游标'],[slider, 360, '', '游标'],])  # 添加新的游标数据
        return data
    
    return None


# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="Archery Vision", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎯 Archery Vision")
        
        with gr.Tabs() as tabs:
            with gr.Tab("1.视频处理"):
                with gr.Row():
                    user_options = gr.BrowserState({})
                    device_dropdown = gr.Dropdown( label="设备选择", choices=["auto", "cpu", "cuda", "mps"], value="auto", interactive=True)
                    bow_hand = gr.Dropdown( label="持弓手", choices=["left", "right"], value="left", interactive=True)
                    model_dropdown = gr.Dropdown( label="模型选择", choices=["yolo11x-pose"], value="yolo11x-pose", interactive=True)
                    batch_size = gr.Number(label="Batch Size", value=8, minimum=1, maximum=64, step=2, precision=0, interactive=True)
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(label="上传视频", sources="upload", interactive=True)
                    with gr.Column():
                        output_video = gr.Video(label="分析结果", format="mp4", interactive=False)
                with gr.Row():         
                    process_btn = gr.Button("开始分析", variant="primary")
                with gr.Row():         
                    status_text = gr.Textbox(label="处理状态", interactive=False, value="等待上传视频...")  # todo 处理完成后可下载csv数据

            with gr.Tab("2.数据分析"):
                with gr.Row():
                    current_frame = gr.Image(label="当前帧", type="numpy", interactive=False)  # todo 图片高度可调整
                with gr.Row():
                    slider = gr.Slider(minimum=0, maximum=100, value=5, step=1, label="拖动滑块移动游标", interactive=True)  # todo 点击按钮按帧前后移动
                with gr.Row():
                    arm_plot = gr.BarPlot(label="双臂姿态角", x="帧号", y="角度", color='动作环节', width=500, height=300)
                    angular_velocity_plot = gr.LinePlot(label="角速度", x="帧号", y="角速度", color='定位', width=500, height=300)
                with gr.Row():
                    angular_acceleration_plot = gr.LinePlot(label="角加速度", x="帧号", y="角加速度", color='定位', width=500, height=300)
                    phase_plot = gr.ScatterPlot(label="相位图", x="角度", y="角速度", color='定位', width=500, height=300)
            
        slider.change(
            fn=Video.extract_frame,inputs=[output_video, slider],outputs=[current_frame]
        ).then(
            fn=update_cursor, inputs=[arm_plot, slider], outputs=[arm_plot]
        ).then(
            fn=update_cursor, inputs=[angular_velocity_plot, slider], outputs=[angular_velocity_plot]
        ).then(
            fn=update_cursor, inputs=[angular_acceleration_plot, slider], outputs=[angular_acceleration_plot]
        ).then(
            fn=update_cursor, inputs=[phase_plot, slider], outputs=[phase_plot]
        )

        process_btn.click(
          fn=lambda user_options, x: user_options.update({'device_dropdown': x}), inputs=[user_options, device_dropdown], outputs=[user_options]
        ).then(
          fn=lambda user_options, x: user_options.update({'bow_hand': x}), inputs=[user_options, bow_hand], outputs=[user_options]
        ).then(
          fn=lambda user_options, x: user_options.update({'model_dropdown': x}), inputs=[user_options, model_dropdown], outputs=[user_options]
        ).then(
          fn=lambda user_options, x: user_options.update({'batch_size': x}), inputs=[user_options, batch_size], outputs=[user_options]
        ).then(
            fn=process_video,
            inputs=[input_video, user_options],
            outputs=[status_text, output_video, slider, current_frame, arm_plot, angular_velocity_plot, angular_acceleration_plot, phase_plot]
        )
    # todo 头部姿态角 头部与脊柱的夹角：
    # 关键错误姿势示例
    # 前肩耸肩（肩角 < 150°）：导致肩部疲劳，箭着点偏低。
    # 后肘塌陷（肘角 < 100°）：力量分散，撒放不干脆。
    # 脊柱侧倾（左右倾斜 > 5°）：影响瞄准一致性，长期导致腰背劳损。
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
