from easydict import EasyDict

config = EasyDict()

#config.openai_api_base = "http://localhost:8000/v1"
#config.model_name = "LungGPT-Dx"
config.openai_api_base = "http://10.9.87.41:8000/v1"
config.model_name = "Qwen2-72B-Instruct-AWQ"
config.model_temperature = 0.7
config.model_top_p = 0.8
config.prompt_version = "v1" 
# v1 provide diseases to choice; v2 let llm to diagnosis; v3 nothing
config.data_path = "data/data.json"
