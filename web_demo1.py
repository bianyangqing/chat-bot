from transformers import AutoModel, AutoTokenizer
import gradio as gr
import logging
import openai

import pinecone

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

EMBEDDINGS_MODEL = "text-embedding-ada-002"
GENERATIVE_MODEL = "gpt-3.5-turbo" # use gpt-4 for better results
EMBEDDING_DIMENSIONS = 1536
TEXT_EMBEDDING_CHUNK_SIZE = 200


PINECONE_API_KEY = "296da2b9-5df6-4d2a-8b77-058137e16a56"
PINECONE_INDEX = "demoindex1"  # dimensions: 1536, metric: cosine similarity
PINECONE_ENV = "us-east4-gcp"
PINECONE_NAMESPACE = "demo_v5"
TOP_K=10


logging.basicConfig(level=logging.INFO)

def load_pinecone_index() -> pinecone.Index:
    """
    Load index from Pinecone, raise error if the index can't be found.
    """
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = PINECONE_INDEX
    if not index_name in pinecone.list_indexes():
        print(pinecone.list_indexes())
        raise KeyError(f"Index '{index_name}' does not exist.")
    index = pinecone.Index(index_name)

    return index


pinecone_index = load_pinecone_index()

def get_embedding(text):
    logging.warning("API key:{}".format(openai.api_key))
    logging.warning("base:{}".format(openai.api_base))
    logging.warning("get_embedding text:{}".format(text))
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    # logging.warning("openai.ChatCompletion.create:{}".format(resp))
    # return openai.Engine(id=engine).embeddings(input=[text])["data"][0]["embedding"]
    return resp['data'][0]['embedding']

def query_knowledge(question, pinecone_index):

    search_query_embedding = get_embedding(question, EMBEDDINGS_MODEL)
    logging.info(f"embedding for question: {search_query_embedding}")

    query_response = pinecone_index.query(
        namespace=PINECONE_NAMESPACE,
        top_k=TOP_K,
        include_metadata=True,
        vector=search_query_embedding,
    )
    logging.warning(
        f"[get_answer_from_files] received query response from Pinecone: {query_response}"
    )

    result = ""
    knowledge_list = query_response['matches']
    if len(knowledge_list) > 0:
        for i in range(min(len(knowledge_list), 2)):
            result = result + "提示{}:".format(i) + knowledge_list[i]['metadata']['content'] + "\n"

    return result



def predict(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []

    knowledge_info = query_knowledge(input)

    query_template = "请严格根据提示回答问题。\n{}问题：{}".format(knowledge_info, input)

    history.append("问题：{}".format(input))

    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="User：" + query))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
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
    button.click(predict, [txt, max_length, top_p, temperature, state], [state] + text_boxes)
demo.queue().launch(share=True, inbrowser=True)