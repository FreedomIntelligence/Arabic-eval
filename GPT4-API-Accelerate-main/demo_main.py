from BotManager import BotManager

if __name__ == '__main__':
    bot_manager = BotManager()
    bot_manager.set_api_key(api_file="api-key.txt", index=0)
    bot_manager.set_proxy(proxy_file="proxy.txt", index=0)
    bot_manager.set_model(api_file="model.txt", index=0)
    bot_manager.set_base_url(api_file="base-url.txt", index=0)
    bot_manager.read_sample(file_name="vicuna_toGPT.jsonl", start=0, end=80)
    bot_manager.set_result_output_dir(result_output_dir="vicuna80_noprompt/")
    bot_manager.multi_process()
    bot_manager.merge_files("vicuna80_noprompt/outputnew1.jsonl")