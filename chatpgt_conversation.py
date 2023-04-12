import openai
import os

import logging
import gradio as gr

the_key_you_need = os.environ.get('the_key_you_need')
openai_api_base = os.environ.get('openai_api_base')
openai.api_key = the_key_you_need
openai.api_base = openai_api_base

prompt = """你是一个智能客服，可以帮助中国的餐饮店老板，在饿了么外卖平台上更好的经营"""
query_template = "请严格根据提示回答问题。如果根据提示无法回答请返回：'抱歉，我的饿了么知识库还在补充中，暂时没有找到相关知识！'\n{}问题：{}"


class ChatGpt:
    def __init__(self, knowledge):
        self.messages = []
        self.knowledge = knowledge
        self.messages.append({"role": "system", "content": prompt})

    def stream_chat(self, question, history=None, box_size=20):

        knowledges = self.knowledge.query_knowledge(question=question, top_k=3)
        query_template.format(knowledges, question)

        logging.warning("before,input:{},history:{}".format(input, history))
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
            updates.append(gr.update(visible=True, value="User：" + input))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + content))
            history[0] = (input, content)
            if len(updates) < box_size:
                updates = updates + [gr.Textbox.update(visible=False)] * (box_size - len(updates))
            logging.warning("result:{}".format([history] + updates))
            yield [history] + updates
