import openai
import gradio as gr

# 设置OpenAI API的访问密钥
openai.api_key = "YOUR_API_KEY"

# 定义聊天接口
def chat(prompt):
    # 调用OpenAI API进行聊天
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
        stream=True  # 开启流式响应
    )
    # 逐步读取响应中的数据
    message = ""
    for chunk in response:
        if "choices" in chunk:
            message += chunk["choices"][0]["text"]
            # 实时更新Gradio界面上的文本框
            gr.interface.set_output(message)
    return message.strip()

# 创建Gradio界面
chat_interface = gr.Interface(
    fn=chat,
    inputs=gr.inputs.Textbox(label="输入"),
    outputs=gr.outputs.Textbox(label="输出", output_type="text", live=True)  # 设置输出参数的live=True
)

# 运行Gradio界面
chat_interface.launch()
