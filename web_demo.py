from transformers import AutoModel, AutoTokenizer
import gradio as gr
import logging

logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def notSupport(model_name, input):
    return "目前不支持:{}".format(model_name)


def predict_by_chatgml(input, max_length, top_p, temperature, model_name, history=[]):
    global updates
    if history is None:
        history = []
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(response)
    return updates[-1]



def predict(input, max_length, top_p, temperature, model_name, history=[]):
    logging.warning("history:{}".format(history))
    logging.warning("input:{}".format(input))
    history.append(input)

    if model_name == "ChatGLM-6B":
        response = predict_by_chatgml(input, max_length, top_p, temperature, model_name, history)
    elif model_name == "chatGpt-api":
        response = notSupport(model_name, input)
    else:
        response = notSupport(model_name, input)

    history.append(response)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history


with gr.Blocks(css="#chatbot{height:350px} .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum lengthhhh", interactive=True)
        top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
        model_name = gr.inputs.Radio(["ChatGLM-6B", "chatGpt-api", "aliXX"], label="Model")
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    txt.submit(predict, [txt, max_length, top_p, temperature, model_name, state], [chatbot, state])
demo.launch(share=True, inbrowser=True)
