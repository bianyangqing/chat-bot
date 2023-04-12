import gradio as gr
import knowledge
import logging
import openai

MAX_BOXES = 20
k = knowledge.KnowledgeMixin()

logging.basicConfig(level=logging.INFO)
query_template = "请严格根据提示回答问题。如果根据提示无法回答请返回：'抱歉，我的饿了么知识库还在补充中，暂时没有找到相关知识！'\n{}问题：{}"


def stream_chat(question, history=None, box_size=20):
    knowledges = k.query_knowledge(question=question, top_k=3)
    query_template.format(knowledges, question)

    logging.warning("before,input:{},history:{}".format(question, history))
    if history is None:
        history = []
    history.append(("", ""))
    messages_copy = [{"role": "user", "content": query_template}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_copy,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
        stream=True
    )
    content = ""
    logging.warning("messageStream0")
    for message in response:
        updates = []
        logging.warning("messageStream0")
        logging.warning("messageStream:{}".format(message))
        if "content" in message['choices'][0]["delta"]:
            delta_content = message['choices'][0]["delta"]["content"]
            delta_content = "" if delta_content is None else delta_content
            content = content + delta_content
        logging.warning("content:{}".format(content))
        updates.append(gr.update(visible=True, value="User：" + question))
        updates.append(gr.update(visible=True, value="ChatGLM-6B：" + content))
        history[0] = (question, content)
        if len(updates) < box_size:
            updates = updates + [gr.Textbox.update(visible=False)] * (box_size - len(updates))
        logging.warning("result:{}".format([history] + updates))
        yield [history] + updates

with gr.Blocks() as demo:
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="Ask a Question："))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="Reply："))

    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=3).style(
                container=False)
        with gr.Column(scale=1):
            button = gr.Button("Generate")
    button.click(stream_chat, [txt, state], [state] + text_boxes)
demo.queue().launch(share=True, inbrowser=True)
