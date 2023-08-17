import openai
import string
import jsonlines

class PostRobot:
    def __init__(self):
        self.api_key = None
        self.proxy = None
        self.model_name = "gpt-3.5-turbo"
        self.role = None
        self.base_url = "https://api.chatanywhere.cn/v1"

    def set_role(self, role):
        if self.role is None:
            self.role = {
                "role": "system",
                "content": role,
            }

    def set_thinking_engine(self, openai_key=None, proxy=None):
        self.set_openai_key(openai_key)
        self.set_proxy(proxy)

    def set_openai_key(self, apikey):
        self.api_key = apikey

    def set_proxy(self, proxy):
        self.proxy = proxy

    def request_chatgpt(self, parameters):
        openai.api_key = self.api_key
        openai.api_base = self.base_url
        response = openai.ChatCompletion.create(
            model=parameters["model"],
            messages=parameters["messages"],
            stream=parameters["stream"]  # this time, we set stream=True
        )
        content=""
        for chunk in response:
            try:
                content+=chunk['choices'][0]['delta']["content"]
            except:
                pass
        return content


    def get_prompt(self, sample):
        text = ""
        if len(sample["instruction"]) > 0 and len(sample["input"]) > 0:
            text = sample["instruction"] + "\n" + sample["input"]
        elif len(sample["instruction"]) > 0 and len(sample["input"]) == 0:
            text = sample["instruction"]
        elif len(sample["instruction"]) == 0 and len(sample["input"]) > 0:
            text = sample["input"]
        return text

    def generate(self, new_message):
        messages = []
        if self.role is not None:
            messages.append(self.role)
        messages.append({"role": "user", "content": new_message})
        parameters = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }
        response=self.request_chatgpt(parameters)
        return True, response


robot = PostRobot()
robot.base_url = "https://api.chatanywhere.cn/v1"
robot.model_name = "gpt-4-0314"
robot.set_thinking_engine("sk-PnkXqX5jMGgrSWRGnaIVmkW6Z7MZ47YgN0QpUNiPThOPhWOQ")
prompt = robot.get_prompt({"instruction": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.Please to answer following question with Arabic.This is the question:في عالم يتزايد الأتمتة فيه، هل من الأهم تحقيق الأولوية لخلق فرص عمل أم التقدم التكنولوجي؟", "input": ""})
flag, response = robot.generate(prompt)
with jsonlines.open('tt.jsonl', mode='a') as writer:
    w ={}
    w["instruction"] = response
    writer.write(w)
# l = list(response)
# l.reverse()
# response1 = "".join(l)
# print(response1)
# sep_response1 = response1.split(' ')
# for i in (sep_response1):
#     print(i)