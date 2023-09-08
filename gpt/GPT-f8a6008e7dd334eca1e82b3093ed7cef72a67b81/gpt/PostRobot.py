import json
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PostRobot:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.history_message = []

    def request_chatgpt(self, openai_key, parameters, new_version=None):
        url = "https://api.openai.com/v1/chat/completions"
        url_1 = "https://43.153.20.180/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        }

        if new_version=="0.1.0":
            url = url_1
        raw_response = requests.post(url, headers=headers, json=parameters, verify=False)
        response = json.loads(raw_response.content.decode("utf-8"))

        try:
            content = response["choices"][0]["message"]["content"]
            flag = True
        except:
            content = response["error"]["code"]
            flag = False
        return flag, content

    def generate(self, api_key, new_message, role=None, args=None, new_version=None):
        if role is not None:
            role = {
                "role": "system",
                "content": role,
            }
        if len(self.history_message) == 0 and role is not None:
            self.history_message.append(role)
        temp_message = self.history_message
        temp_message.append({"role": "user", "content": new_message})
        parameters = {
            "model": self.model_name,
            "messages": temp_message
        }
        if args is not None:
            for key in args.keys():
                parameters[key] = args[key]
        flag, response = self.request_chatgpt(api_key, parameters, new_version)
        if flag == True:
            self.history_message.append({"role": "user", "content": new_message})
            self.history_message.append({"role": "assistant", "content": response})
        return flag, response
