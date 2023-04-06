import gradio as gr
import logging

logging.basicConfig(level=logging.INFO)

def predict(input, history=[]):
    logging.warning("history:{}".format(history))
    history.append(input)
    response = "收到了一个问题：{}".format(input)
    history.append(response)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history


with gr.Blocks(css="#chatbot{height:350px} .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    txt.submit(predict, [txt, state], [chatbot, state])

demo.launch(share=True,inbrowser=True)