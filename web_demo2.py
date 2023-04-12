import time

from transformers import AutoModel, AutoTokenizer
import gradio as gr
import openai
import os

import logging


the_key_you_need = os.environ.get('the_key_you_need')

openai_api_base = os.environ.get('openai_api_base')


logging.basicConfig(level=logging.INFO)


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2



tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()
def predict(input, max_length, top_p, temperature, history=None):
    logging.warning("before,input:{},history:{}".format(input, history))
    if history is None:
        history = []

    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="User：" + query))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        logging.warning("after,updates:{},history:{}".format(updates, history))
        logging.warning("yield_result:{}".format([history] + updates))
        yield [history] + updates


def predictByGpt(input, max_length, top_p, temperature, history=None):
    openai.api_key = the_key_you_need
    openai.api_base = openai_api_base
    logging.warning("before,input:{},history:{}".format(input, history))
    if history is None:
        history = []
    history.append(("",""))

    messages_copy = [{"role": "user", "content": input}]
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
            updates.append(gr.update(visible=True, value="User：" + input))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + content))
            history[0] = (input, content)
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates


def predictTest(input, max_length, top_p, temperature, history=None):
    history = []
    history.append(("",""))
    content = ""
    for i in range(100):
        updates = []
        content = content + str(i)
        updates.append(gr.Textbox.update(visible=True, value="User：" + input))
        updates.append(gr.Textbox.update(visible=True, value="ChatGLM-6B：" + content))
        # logging.warning("in:{}".format(content))
        history[0] = (input, content)
        time.sleep(0.1)
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
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
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=11).style(
                container=False)
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            button = gr.Button("Generate")
    button.click(predictTest(), [txt, max_length, top_p, temperature, state], [state] + text_boxes)
demo.queue().launch(share=True, inbrowser=True)