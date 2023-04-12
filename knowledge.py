import logging
import openai
import pinecone

logging.basicConfig(level=logging.INFO)


class KnowledgeMixin:
    def __init__(self):
        self.EMBEDDINGS_MODEL = "text-embedding-ada-002"
        self.PINECONE_API_KEY = "296da2b9-5df6-4d2a-8b77-058137e16a56"
        self.PINECONE_INDEX = "demoindex1"
        self.PINECONE_ENV = "us-east4-gcp"
        self.PINECONE_NAMESPACE = "demo_v5"
        self.index = self.load_pinecone_index()

    def load_pinecone_index(self):
        """
                    Load index from Pinecone, raise error if the index can't be found.
                    """
        pinecone.init(
            api_key=self.PINECONE_API_KEY,
            environment=self.PINECONE_ENV,
        )
        index_name = self.PINECONE_INDEX
        if not index_name in pinecone.list_indexes():
            print(pinecone.list_indexes())
            raise KeyError(f"Index '{index_name}' does not exist.")
        index = pinecone.Index(index_name)
        return index

    def get_embedding(self, text):
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

    def query_knowledge(self, question, top_k=3):

        search_query_embedding = self.get_embedding(question)
        logging.info(f"embedding for question: {search_query_embedding}")

        query_response = self.index.query(
            namespace=self.PINECONE_NAMESPACE,
            top_k=top_k,
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
