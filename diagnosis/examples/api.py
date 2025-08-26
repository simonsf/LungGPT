import os
import json

import sys
project_dir = '/data/yhb/code/LungGPT'
sys.path.append(project_dir)

from openai import OpenAI
from config.config import config as cfg

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = cfg.openai_api_base

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def get_prompt(version="v1"):
    if version == "v1":
        start_prompt = '''请根据以下"输入"中病人的信息, 在“疾病列表”中选择患者可能患有的疾病，输出只包含疾病列表中的名称，不要输出诊断原因和建议。{"疾病列表":["上呼吸道感染","急性上呼吸道感染","细菌性肺炎","曲霉菌性肺炎","继发性肺结核","陈旧性肺结核","肺结核","病毒性肺炎","重症肺炎","肺部感染","肺炎","阻塞性肺炎","支气管哮喘","哮喘","支气管扩张伴咯血","支气管哮喘 (急性发作期)","慢性阻塞性肺病","慢性阻塞性肺病伴有急性加重","慢性阻塞性肺病半有急性下呼吸道感染","支气管扩张(症)","支气管扩张伴感染","肺沛源性心脏病","慢性肺源性心脏病","I型呼吸衰竭","II型呼吸衰竭","呼吸衰竭","急性呼吸窘迫综合征","右肺上叶恶性肿瘤","肺恶性肿瘤","左肺上叶恶性肿瘤","右肺下叶恶性肿瘤","右肺恶性肿瘤","左肺下叶恶性肿瘤","左肺恶生肿瘤","右肺中叶恶性肿瘤","肺肿瘤","肺栓塞","急性肺血栓栓塞症","急性肺栓塞","肺动脉高压","肺动脉高压中度","肺动脉高压重度","肺水肿"","间质性肺病","肺间质改变",间质性肺炎","肺间质纤维化"","药物性间质性肺疾患","尘肺","矽肺",肺挫伤","肺损伤","放射性肺炎","吸入性肺炎","睡眠呼吸暂停低通气综合征","阻塞性睡眠呼吸暂停低通气综合征","胸腔积液","恶性胸腔积液","气胸","自发性气胸","液气胸","血气胸","纵隔淋巴结继发恶性肿瘤","纵隔淋巴结肿大","孤立性肺结节
        ","肺气肿合并肺大泡","肺大疱","阻塞性肺气肿","肺气肿"],"输入":'''
        end_prompt = '''}'''
    elif version == "v2":
        start_prompt = '''请根据以下"输入"中病人的信息, 做出诊断,输出只包含疾病名称，不要输出诊断原因和建议。{"输入":'''
        end_prompt = '''}'''
    else:
        start_prompt = ''''''
        end_prompt = ''''''
    return start_prompt, end_prompt

def predict(message, history):
    # Convert chat history to OpenAI format
    history_openai_format = [{
        "role": "system",
        "content": "You are helpful assistant"
    }]
    history_openai_format.append({"role": "user", "content": message})

    # Create a chat completion request and send it to the API server

    stream = client.chat.completions.create(
        model=cfg.model_name,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=cfg.model_temperature ,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            'repetition_penalty':
                1.05,
            'stop_token_ids': [],
            'top_p': cfg.model_top_p
        })

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
    
    return partial_message

if __name__ == "__main__":
    with open(os.path.join(project_dir,cfg.data_path)) as f:
      sample_list = json.load(f)
    start_prompt, end_prompt = get_prompt(cfg.prompt_version)
    for sample in sample_list:
      message = start_prompt + sample["input"] + end_prompt
      try:
          print("query:", message)
          response = predict(message, "")
          print("response:", response)
          print("groudtruch:", sample["output"])
      except Exception as e:
          print(e)

