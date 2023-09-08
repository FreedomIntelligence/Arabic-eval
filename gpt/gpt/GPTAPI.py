import configparser
import requests

from gpt.PostRobot import PostRobot
import os


class GPT:
    def __init__(self, user_name=None, model_name="gpt-3.5-turbo", new_version=None):
        self.url = self.get_address()
        self.user_name = user_name
        self.post_robot = PostRobot(model_name)
        self.new_version = new_version

    def get_address(self):
        # 创建一个 ConfigParser 实例
        config = configparser.ConfigParser()
        current_path, _ = os.path.split(os.path.abspath(__file__))
        config.read(os.path.join(current_path, 'apikey.config'))
        # 获取配置项的值
        ip = config.get('Address', 'IP')
        port = config.get('Address', 'Port')
        return ip + ":" + port + "/"

    def get_random_line(self, username):
        url = self.url + "?username=" + username
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("api-key")
        else:
            print("Error:", response.status_code)
            return None

    def call(self, new_message, role=None, args=None):
        api_key = self.get_random_line(username=self.user_name)
        if api_key is not None:
            return self.post_robot.generate(api_key, new_message, role, args, new_version=self.new_version)
        else:
            return False, "APIKey Error"


class GPTAgent:
    def __init__(self):
        self.url = self.get_address()

    def get_address(self):
        # 创建一个 ConfigParser 实例
        config = configparser.ConfigParser()
        current_path, _ = os.path.split(os.path.abspath(__file__))
        config.read(os.path.join(current_path, 'apikey.config'))
        # 获取配置项的值
        ip = config.get('AgentAddress', 'IP')
        port = config.get('AgentAddress', 'Port')
        return ip + ":" + port + "/gpt"

    def call(self, new_message, role="", args=None):
        if args is None:
            args = {}
        data = {
            'new_message': new_message,
            'role': role,
            'args': args
        }
        response = requests.post(self.url, json=data)
        result = response.json()
        return result


if __name__ == '__main__':
    # if you need to call in bulk for a long time, just use your computer to request OpenAI API with the VPN.
    gpt = GPT(user_name="zhangsan")
    # gtp=GPT(user_name="zhangsan",model_name="gpt-3.5-turbo-16k")
    flag, response = gpt.call("今天肚子很饿")
    if flag == True:
        print(response)
    else:
        print(f'error: {response}')
    # if you don't have the VPN and don't need to call in bulk for a long time, you can ask the server to request OpenAI API.
    # gpt = GPTAgent()
    # flag, response = gpt.call("今天肚子很饿")
    # if flag == True:
    #     print(response)
    # else:
    #     print(f'error: {response}')
