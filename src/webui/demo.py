from random import randint
import gradio as gr
import pandas as pd

# 创建带有游标线的折线图
def create_plot_with_cursor(cursor_pos):
    data = pd.DataFrame({
            "time": list(i for i in range(200)) + [cursor_pos, cursor_pos],
            "temperature": [randint(50 + 10 * (i % 2), 65 + 15 * (i % 2)) for i in range(200)] + [0, 80],
            "location": ["indoor"] * 200 + ["outdoor"] * 2,
        })
    # 创建LinePlot对象
    line_plot = gr.LinePlot(visible=True, value=data, inputs=slider, x="time", y="temperature", color="location")
    return line_plot

# 主界面
with gr.Blocks() as demo:
    gr.Markdown("## 动态游标线折线图示例")
    
    with gr.Row():
        # 拖拽条控制游标位置
        slider = gr.Slider(minimum=0,maximum=200,value=5,step=1,label="游标位置",interactive=True)
        
    with gr.Row():
        # 折线图区域
        plot = gr.LinePlot(visible=True, x="time", y="temperature")

    # 交互逻辑
    slider.change(fn=lambda pos: create_plot_with_cursor(pos),inputs=slider,outputs=plot)
    
    # 初始化图表
    demo.load(fn=lambda: create_plot_with_cursor(5),outputs=plot)

if __name__ == "__main__":
    demo.launch()