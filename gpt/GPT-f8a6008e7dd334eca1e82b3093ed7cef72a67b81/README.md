# 📌GPT
It is a private chatGPT API and you don't need the API-KEY to get the answer from ChatGPT.

# 🆚Compare with OpenAI

Method| API-KEY |VPN| Request Machine | Intensity
---|-----|---|---|---
OpenAI| ✅  |✅| Yours Computer|⚡️⚡️(Limited by the performance of a single account)
GPT(ours)| ❌  |✅ | Yours Computer|⚡️⚡️⚡️(Limited by the performance of your computer)
(Temporarily unavailable at 0.0.9) ~GPTAgent(ours)~| ❌  |❌ | Server|⚡️(Limited by the performance of the server)

# 🚀Update
📢[version 0.1.0] We have released the beta version 0.1.0, but it won't affect the original use of 0.0.9. In this new version, you can use this API as long as you can connect to our school's network. No VPN is needed. To try 0.1.0 out, please add a `new_version=0.1.0` when you instantiate the GPT object. Nothing else is changed. After all, this is only beta version. eg. `gpt = GPT(model_name='gpt-3.5-turbo-16k', user_name='ZhangSan', new_version='0.1.0')`

📢[version 0.0.9] Due to frequent official deactivation on the API, we are attempting to add user signatures to understand our API usage and identify the problem. Therefore, please do not make large-scale and long-term calls on this version (0.0.9) (this version is a test version). If there is an urgent need to use the key as a result, please get in touch with me.

📢[version 0.0.8] We have temporarily switched to a local connection due to network fluctuations. The GPTAgent() function is temporarily unavailable, but the GPT() function is running normally.

📢[version 0.0.7] Now, the GPT support the custom model, including the latest version "gpt-3.5-turbo-16k". The function call is coming soon. 

📢[version 0.0.6] Update the ip.

📢[version 0.0.5] Now, the GPT support multi-turn conversations and each instance is a session.

📢[version 0.0.4] Add the GPTAgent that means you needn't both API-KEY and VPN to use it if you don't need to call in bulk for a long time.

📢[version 0.0.3] Add exception capture in call function.

📢[version 0.0.2] Add the args in the call function.

When calling the model, you can add the args, which is a dictionary of the parameters of OpenAI API, such as top_p, max_tokens, temperature, etc.

## 📖 How to install 
```
pip install git+https://github.com/FreedomIntelligence/GPT.git
```
## 📋 How to use
### Single-turn GPT(): If you need to call in bulk for a long time, just use your computer to request OpenAI API with the VPN.
（🚨**Note**: Since the 0.0.5 version, GPT() has supported multi-turn conversation, which means it has the history of the requests. If you want to use it in single-turn mode, please instantiate a GPT () before each use. ）
```
from gpt import GPT
gpt=GPT(user_name="Your name")
# gpt=GPT(model_name="gpt-3.5-turbo-16k")
flag, response = gpt.call("今天肚子很饿")
if flag == True:
    print(response)
else:
    print(f'error: {response}')
```
### Multi-turn GPT(): If you want a multi-turn conversation.
```
from gpt import GPT
role_a = "你一个去周记牛肉火锅的顾客。"
role_b = "你是周记牛肉火锅的服务员。"
start_sentence = "你好，请问今天吃点什么？"
gpt_a = GPT(user_name="Your name")
gpt_b = GPT(user_name="Your name")
print(f"服务员：{start_sentence}")
flag, response_a = gpt_a.call(start_sentence, role_a)
print(f"顾客：{response_a}")
for i in range(10):
    flag, response_b = gpt_b.call(response_a, role_b)
    print(f"服务员：{response_b}")
    flag, response_a = gpt_a.call(response_b, role_a)
    print(f"顾客：{response_a}")
```

### Using args to constraint generation

coming soon.

### Common Error Causes
1. `error: insufficient quota` : 额度用完了。
2. `This key is associated with a deactivated account.` : 账号被封号了。
3. `reguests.exceptions.ConnectionError: ('Connection aborted.",RemoteDisconnected("Remote end closed connection without response")` : 一般为使用者的VPN不稳定。
4. `<class'requests.exceptions.ConnectionError'.HTTPConnectionPool(host='10.20.12.38' port=5000): Max retries exceeded with url: /?usernae=MyName (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0808023216591340>: Failed to establish a new connection:[WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。”))` : 一般为使用者的VPN不稳定。
5. `请求官方API失败。Error code: 200` : 该错误源于使用者使用了0.1.0的GPT库，这个功能还不完善，请使用0.0.9版。直接使用`How to install`里的命令行安装该库就可以保证正确的版本了。
