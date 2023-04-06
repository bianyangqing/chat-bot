import openai
import os
import gradio as gr
import requests
import json

knowlage = {
    "怎么解绑银行卡？": "解绑银行卡操作如下：手机端：【饿了么商家版】-右下角【我的】-【钱包】-【账户设置】-【提现账户】-解绑银行卡温馨提示：余额不超过100 ，可点击解绑，超过100，不可解绑，只可换绑哦~,电脑端：【饿了么商家版】-左侧【财务管理】-【账户管理】-右侧【解绑】即可。温馨提示：余额不超过100 ，可点击解绑，超过100，不可解绑，只可换绑哦~",
    "怎么查看提现记录？": "查看提现记录步骤如下：电脑端：财务-财务首页—查看流水记录,手机端：商家端-我的-钱包-查看明细，进行查看。",
    "提现规则是什么？": "老板您好，现在提现方法有手动提现和自动提现两种方式。,自动提现规则：,1、每天一次自动提现，开启自动提现后，每日上午6点到8点期间，每日余额超过500元才会自动提现。,2、钱包余额低于500元无法自动提现时，商户可选择去APP或PC端手动提现。,手动提现规则：手动提现设置最低额度，钱包余额超过50元，才可以手动提现。低于50元的钱包余额后续商家闭店后可以将此金额提现到银行卡/支付宝中。,注：默认要求提现人完成实名认证，提现收款银行卡与提现人同名，非同名不能绑卡。,温馨提示：平台提现到账时间为即时到账，一般会在10min左右到达您银行卡，如果您提现没有立即到账，建议您后续关注，预计会在1～3天到账平台提现到账时间为即时到账。"
}

openai.url = "https://{}/v1/chat/completions".format(os.environ.get("OPENAI_API_PROXY_DOMAIN"))
openai.api_key = os.environ.get("OPENAI_API_KEY")

OPENAI_URL = "https://{}/v1/chat/completions".format(os.environ.get("OPENAI_API_PROXY_DOMAIN"))
OPENAI_AI_KEY = os.environ.get("OPENAI_API_KEY")


class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def askGpt(self, question, knowlageStr):
        try:
            self.messages.append({"role": "user", "content": knowlageStr + " " + question})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round * 2 + 1:
            del self.messages[1:3]
        return message

    def askWithProxy(self, question):

        print("start askWithProxy")
        self.messages.append({"role": "user", "content": question})

        headers = {
            'Authorization': f'Bearer {OPENAI_AI_KEY}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": self.messages
                }
            ]
        }
        try:
            response = requests.post(OPENAI_URL, headers=headers, json=payload)
            response.raise_for_status()  # 抛出异常，如果响应码不是200
            data = response.json()
            print("http_response:{}".format(data["choices"][0]["message"]))
            message = data["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": message})
            if len(self.messages) > self.num_of_round * 2 + 1:
                del self.messages[1:3]
            return message
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
        except json.JSONDecodeError as e:
            print(f"无效的 JSON 响应: {e}")


prompt = """你是一个智能客服，可以帮助中国的餐饮店老板，在饿了么外卖平台上更好的经营"""

conv = Conversation(prompt, 5)


def predict(input, history=[]):
    history.append(input)
    response = ""
    if input in knowlage:
        response = conv.askGpt(input, knowlage[input])
    else:
        response = "抱歉！我只能回答一下代码，请复制其中一个问题提问：{}".format('\n'.join(knowlage.keys()))

    history.append(response)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history


with gr.Blocks(css="#chatbot{height:350px} .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    txt.submit(predict, [txt, state], [chatbot, state])

demo.launch(share=False,inbrowser=True)