import os
import openai
api_key = "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"
openai.api_key = api_key
openai.api_base = 'https://openai-proxy-aio.pages.dev/api/v1'
# "https://api.openai.com/v1/embeddings"
resp = openai.Embedding.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter..."
)

print(resp)