import gradio as gr
import knowledge
import logging
import openai
import os

MAX_BOXES = 20
k = knowledge.KnowledgeMixin()

logging.basicConfig(level=logging.INFO)
# query_template = "请严格根据提示回答问题,尽量给出详细的操作步骤。如果根据提示无法回答请返回：'抱歉，我的饿了么知识库还在补充中，暂时没有找到相关知识！'\n{}问题：{}"
query_template = "尽量按照给出的提示回答问题，回答尽量详细具有可操作性，语气尽量热情'\n{}问题：{}"
openai.api_key = os.environ.get('the_key_you_need')
openai.api_base = os.environ.get('openai_api_base')

NOTE_TXT = "试试以下问题吧：\n" \
           "1、怎么发布对用户有吸引力的商品？" \
           "2、如何提升曝光量？" \
           "3、如何提升店铺质量分？" \
           "4、店铺下线了，如何处理？" \
           "5、店铺二维码怎么分享？" \
           "版本说明：\n" \
           "1、大模型接口调用较慢\n" \
           "2、目前饿了么商家知识库数据量还非常小，可能会出现“幻觉”现象并返回不符合事实的信息（尤其是页面配置、url等）\n" \
           "3、V2版本计划提升回答的准确性、其他用户体验等。\n" \



def stream_chat(question, history=None, box_size=20):
    logging.warning("before,input:{},history:{}".format(question, history))

    knowledges = k.query_knowledge(question=question, top_k=6)
    question_with_template = query_template.format(knowledges, question)

    if history is None:
        history = []
    messages_copy = [{"role": "user", "content": question_with_template}]
    logging.warning("before,message{}".format(messages_copy))

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_copy,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
        stream=True
    )
    content = ""
    for message in response:
        if len(content)>0 and len(history)>0:
            # history中去掉最后一个元素
            history.pop()

        updates = []
        if "content" in message['choices'][0]["delta"]:
            delta_content = message['choices'][0]["delta"]["content"]
        else:
            delta_content = " "
        delta_content = " " if delta_content is None else delta_content
        content = content + delta_content
        logging.warning("content:{}".format(content))
        history.append((question, content))
        for question1, content1 in history:
            updates.append(gr.update(visible=True, value="用户：" + question1))
            updates.append(gr.update(visible=True, value="智能助手：" + content1))
        if len(updates) < box_size:
            updates = updates + [gr.Textbox.update(visible=False)] * (box_size - len(updates))
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

            note = gr.Textbox(show_label=False,
                             placeholder=NOTE_TXT,
                             lines=5,
                             interactive=False).style(container=False)

            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=3).style(
                container=False)
        with gr.Column(scale=1):
            button = gr.Button("Generate")
    button.click(stream_chat, [txt, state], [state] + text_boxes)
demo.queue().launch(share=True, inbrowser=True)
