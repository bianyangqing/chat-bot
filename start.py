import gradio as gr
import knowledge
import chatpgt_conversation
import logging

MAX_BOXES = 20
k = knowledge.KnowledgeMixin()
chatgpt = chatpgt_conversation.ChatGpt(k)

logging.basicConfig(level=logging.INFO)

def predict(input, history=None):
    logging.warning("question_received:{}".format(input))
    return chatgpt.stream_chat(input, history, MAX_BOXES)


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
    button.click(predict, [txt, state], [state] + text_boxes)
demo.queue().launch(share=True, inbrowser=True)
