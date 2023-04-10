import requests
import json
import os

url = "https://chatgptproxyapi-atq.pages.dev/v1/chat/completions"
api_key = os.environ.get('VARIABLE_NAME')


headers = {
  'Authorization': f'Bearer {api_key}',
  'Content-Type': 'application/json'
}

payload = {
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "what is chatgpt"
    }
  ]
}


try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status() # 抛出异常，如果响应码不是200
    data = response.json()
    print(data)
except requests.exceptions.RequestException as e:
    print(f"请求错误: {e}")
except json.JSONDecodeError as e:
    print(f"无效的 JSON 响应: {e}")