from transformers import AutoModel, AutoTokenizer
import gradio as gr
import logging
import openai
import pinecone
import uuid
import random
import os
import copy

NOTE = 'V1版本局限性说明 \n' \
       '1、大模型接口调用较慢，目前流式打字机效果还在调试中，请耐心等待。（如果我们明天打字机出不来的话）\n' \
       '2、目前饿了么商家知识库数据量还非常小，可能会出现“幻觉”现象并返回不符合事实的信息（尤其是页面配置、url等）。\n' \
       '3、V2版本计划提升回答的准确性、其他用户体验等。'

prompt = """你是一个智能客服，可以帮助中国的餐饮店老板，在饿了么外卖平台上更好的经营"""

session_id = str(uuid.uuid4().hex)

logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()
chatgml_chat_history = []

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

EMBEDDINGS_MODEL = "text-embedding-ada-002"
GENERATIVE_MODEL = "gpt-3.5-turbo" # use gpt-4 for better results
EMBEDDING_DIMENSIONS = 1536
TEXT_EMBEDDING_CHUNK_SIZE = 200
TOP_K = 20
COSINE_SIM_THRESHOLD = 0.7
MAX_TEXTS_TO_EMBED_BATCH_SIZE = 100
MAX_PINECONE_VECTORS_TO_UPSERT_PATCH_SIZE = 100


PINECONE_API_KEY = "296da2b9-5df6-4d2a-8b77-058137e16a56"
PINECONE_INDEX = "demoindex1"  # dimensions: 1536, metric: cosine similarity
PINECONE_ENV = "us-east4-gcp"
PINECONE_NAMESPACE = "demo_v5"




url = "https://195.245.242.82:8443/v1/chat/completions"
the_key_you_need = os.environ.get('the_key_you_need')

openai_api_base = os.environ.get('openai_api_base')

headers = {
  'Authorization': f'Bearer {the_key_you_need}',
  'Content-Type': 'application/json'
}

payload_chat = {
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "what is chatgpt"
    }
  ]
}




def add_random_chars(string):
    result = ""
    for char in string:
        random_char = chr(random.randint(33, 126))  # 生成33到126之间的随机ASCII码
        result += char + random_char
    return result

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

def get_embedding(text, engine):
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

def query_knowledge(question, session_id, pinecone_index):

    search_query_embedding = get_embedding(question, EMBEDDINGS_MODEL)
    logging.info(f"embedding for question: {search_query_embedding}")

    query_response = pinecone_index.query(
        namespace=PINECONE_NAMESPACE,
        top_k=TOP_K,
        include_metadata=True,
        vector=search_query_embedding,
    )
    # logging.warning(
    #     f"[get_answer_from_files] received query response from Pinecone: {query_response}"
    # )

    result = ""
    knowledge_list = query_response['matches']
    if len(knowledge_list) > 0:
        for i in range(min(len(knowledge_list), 2)):
            result = result + "提示{}:".format(i) + knowledge_list[i]['metadata']['content'] + "\n"

    return result



def get_answer_from_files(question, session_id, pinecone_index):
    logging.info(f"Getting answer for question: {question}")

    search_query_embedding = get_embedding(question, EMBEDDINGS_MODEL)
    logging.info(f"embedding for question: {search_query_embedding}")

    try:
        query_response = pinecone_index.query(
            namespace=session_id,
            top_k=TOP_K,
            include_values=False,
            include_metadata=True,
            vector=search_query_embedding,
        )
        logging.warning(
            f"[get_answer_from_files] received query response from Pinecone: {query_response}"
        )

        files_string = ""
        # file_text_dict = current_app.config["file_text_dict"]

        file_string = ""
        for i in range(len(query_response.matches)):
            result = query_response.matches[i]
            file_chunk_id = result.id
            score = result.score
            filename = result.metadata["filename"]
            # file_text = file_text_dict.get(file_chunk_id)
            # file_string = f"###\n\"{filename}\"\n{file_text}\n"
            if score < COSINE_SIM_THRESHOLD and i > 0:
                logging.info(
                    f"[get_answer_from_files] score {score} is below threshold {COSINE_SIM_THRESHOLD} and i is {i}, breaking")
                break
            files_string += file_string

        # Note: this is not the proper way to use the ChatGPT conversational format, but it works for now
        messages = [
            {
                "role": "system",
                "content": f"Given a question, try to answer it using the content of the file extracts below, and if you cannot answer, or find " \
                           f"a relevant file, just output \"I couldn't find the answer to that question in your files.\".\n\n" \
                           f"If the answer is not contained in the files or if there are no file extracts, respond with \"I couldn't find the answer " \
                           f"to that question in your files.\" If the question is not actually a question, respond with \"That's not a valid question.\"\n\n" \
                           f"In the cases where you can find the answer, first give the answer. Then explain how you found the answer from the source or sources, " \
                           f"and use the exact filenames of the source files you mention. Do not make up the names of any other files other than those mentioned " \
                           f"in the files context. Give the answer in markdown format." \
                           f"Use the following format:\n\nQuestion: <question>\n\nFiles:\n<###\n\"filename 1\"\nfile text>\n<###\n\"filename 2\"\nfile text>...\n\n" \
                           f"Answer: <answer or \"I couldn't find the answer to that question in your files\" or \"That's not a valid question.\">\n\n" \
                           f"Question: {question}\n\n" \
                           f"Files:\n{files_string}\n" \
                           f"Answer:"
            },
        ]

        response = openai.ChatCompletion.create(
            messages=messages,
            model=GENERATIVE_MODEL,
            max_tokens=1000,
            temperature=0,
        )

        choices = response["choices"]  # type: ignore
        answer = choices[0].message.content.strip()

        logging.info(f"[get_answer_from_files] answer: {answer}")

        return answer

    except Exception as e:
        logging.info(f"[get_answer_from_files] error: {e}")
        return str(e)


class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, query_origin, question_template):
        logging.warning("gpttask_key:{}".format(openai.api_key))
        logging.warning("gptbase:{}".format(openai.api_base))
        try:

            messages_copy = []
            messages_copy.append({"role": "user", "content": question_template})
            self.messages.append({"role": "user", "content": query_origin})

            logging.warning("gpt_messages withStream:{}".format(messages_copy))
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages_copy,
                temperature=0.5,
                max_tokens=2048,
                top_p=1
            )

        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]

        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round * 2 + 1:
            del self.messages[1:3]
        return message


conv = Conversation(prompt, 5)


def notSupport(model_name, input):
    return "目前不支持:{}".format(model_name)


def predict_by_chatgml(input, max_length, top_p, temperature, model_name, apikey , history=None):
    if history is None:
        history = []
    global updates
    if history is None:
        history = []
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(response)
    return updates[-1]


def predict(input, model_name,  history=None):
    openai.api_key = the_key_you_need

    openai.api_base = openai_api_base

    logging.warning("history:{}".format(history))
    logging.warning("openai.api_key:{}".format(openai.api_key))
    logging.warning("input:{}".format(input))
    logging.warning("model_name:{}".format(model_name))

    knowledge_info = query_knowledge(input, session_id, pinecone_index)

    query_template = "请严格根据提示回答问题。如果根据提示无法回答请返回：'抱歉，我的饿了么知识库还在补充中，暂时没有找到相关知识！'\n{}问题：{}".format(knowledge_info, input)

    history.append("问题：{}".format(input))

    logging.warning("query_template:{}".format(query_template))
    if model_name == "ChatGLM-6B":
        response = predict_by_chatgml(query_template, 2048, 0.7, 0.95, model_name, chatgml_chat_history)
    elif model_name == "chatGpt-api":
        response = conv.ask(input, query_template)
    else:
        response = notSupport(model_name, query_template)

    chatgml_chat_history.append((query_template, response))
    history.append("{}".format(response))

    logging.warning("response:{}".format(response))
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history


with gr.Blocks(css="#chatbot{height:600px} .overflow-y-auto{height:500px}",
               title="商家小助手",
               description= NOTE,
) as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        # max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum lengthhhh", interactive=True)
        # top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
        # temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
        model_name = gr.inputs.Radio(["chatGpt-api", "ChatGLM-6B", "aliXX"],default="chatGpt-api", label="Model")
        # apikey = gr.Textbox(show_label=False, placeholder="Enter chatGpt api key sk-xxxxx").style(container=False)
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", label="Question").style(container=False)

    txt.submit(predict, [txt, model_name, state], [chatbot, state])
demo.launch(share=True, inbrowser=True)

