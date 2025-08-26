import os
import json
import time
import torch
import argparse

from openai import OpenAI
from collections import OrderedDict


class JSONObject:
    def __init__(self, d):
        for key, value in d.items():
            setattr(self, key, value)


class InferenceAPI:
    def __init__(self, cfg_path) -> None:
        self.cfg_path = cfg_path
        self._check_config()
        self._load_data()
        self.out_dict = OrderedDict()
        self.case_count = 0
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.cfg.model_url,
        )
        self.out_fname = os.path.join(self.cfg.save_path, f"{self.inf_task}_result.json")

    def _load_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as rf:
            data = json.load(rf)
        return data

    def _check_config(self):
        self.cfg = JSONObject(self._load_json(self.cfg_path))

    def _load_data(self):
        # some string
        self.admission_ind = self.cfg.addmission_ind
        self.diagnosis_ind = self.cfg.diagnosis_ind
        self.inference_ind = self.cfg.inference_ind if self.cfg.with_inference_ref else ""
        self.shot_ind = self.cfg.shot_ind if self.cfg.with_shot else ""
        self.block_flag = self.cfg.block_flag
        self.with_shot = self.cfg.with_shot
        self.with_inference_ref = self.cfg.with_inference_ref
        # json 
        self.inf_task = self.cfg.inference_task
        self.prompt_path = os.path.join("data", f"diagnosis_{self.inf_task}", f"prompt_{self.inf_task}.json")
        self.shot_path = os.path.join("data", f"diagnosis_{self.inf_task}", f"shot_{self.inf_task}.json")
        self.disease_list = self._load_json(self.cfg.disease_class)['disease']
        self.sample_data = self._load_json(self.cfg.test_data)
        self.disease_dict = self._load_json(self.cfg.bookclass)
        self.inference_dict = self._load_json(self.cfg.reference)["content"]
        self.role_dict = self._load_json(self.cfg.system_role)
        prompt_dict = self._load_json(self.prompt_path)
        self.prompt_str = prompt_dict["prompt_shot"] if self.cfg.with_shot else prompt_dict["prompt"]
        self.shot_str = self._load_json(self.shot_path)["shot"] if self.cfg.with_shot else ""
        # API config
        self.openai_api_key = self.cfg.openai_api_key
        self.base_url = self.cfg.model_url
        self.model = self.cfg.model
        self.temperature = self.cfg.temperature
        self.top_p = self.cfg.top_p
        self.stop_token_ids = self.cfg.stop_token_ids

    def save_json(self, data):
        with open(self.out_fname, 'w', encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))

    def get_result_dict(self, sample, response, dia4reasoning, response_key="诊断依据"):
        key_excluded = ["symptom_raw", "output_raw", "output"]
        final_dict = OrderedDict()
        for key, value in sample.items():
            if key in key_excluded:
                continue
            final_dict[key] = value

        final_dict[response_key] = response
        final_dict["入院诊断"] = dia4reasoning
        return final_dict
    
    def find_value_in_nested_dict(self, data, key):
        """
        在嵌套字典中查找指定key对应的值
        param data: 嵌套字典
        param key: 要查找的key
        Returns:
            找到则返回对应的值，否则返回None
        """

        if isinstance(data, dict):
            for k, v in data.items():
                if k == key:
                    return v
                elif isinstance(v, (dict, list)):
                    result = self.find_value_in_nested_dict(v, key)
                    if result:
                        return result
        return None
  
    def get_common_inference_key(self, class_dict, disease):
        """
        解决一些亚类没有，但有大类的公共治疗方法
        param class_dict: 和诊疗指南吻合的疾病类别字典
        param disease: 当前疾病亚类名
        """
        if isinstance(class_dict, dict):
            for k, v in class_dict.items():
                if isinstance(v, list):
                    if disease in v:
                        return k
                elif isinstance(v, dict):
                    if disease in v.keys():
                        return k
                    result = self.get_common_inference_key(v, disease)
                    if result:
                        return result
        return None

    def get_inference(self, disease):
        """
        解决一些亚类没有，但有大类的公共治疗方法
        param inference_dict: 治病治疗信息字典
        param disease: 当前疾病亚类名
        """
        # 先从治疗字典里找，无返回None
        inference = self.find_value_in_nested_dict(self.inference_dict, disease)
        if inference:
            return inference
        # 返回None情况，从映射表找到名字
        else:
            # 这里common_inference_key里可能没有疾病，返回会是none
            disease = self.get_common_inference_key(self.disease_dict, disease)
            if disease:
                inference = self.get_inference(disease)
                if inference:
                    return inference
        return None

    def inference_dict2str(self, inference_dict):
        """
        Fromating 
        """  
        if isinstance(inference_dict, dict):
            # if list(inference_dict.keys())[0] == "治疗":
            #     inference_dict = inference_dict["治疗"]
            if isinstance(inference_dict, dict):
                return  ",".join(f"{key}: {self.inference_dict2str(value)}" for key, value in inference_dict.items())
            else:
                return inference_dict
        elif isinstance(inference_dict, list):
            return ",".join(self.inference_dict2str(item) for item in inference_dict)
        else:
            return str(inference_dict)
    
    def get_llm_response(self, sample):
        history_openai_format = []
        history_openai_format.append(self.role_dict)
        message, dia4reasoning = self.wrap_input_message(sample)
        history_openai_format.append({"role": "user", "content": message})
        print('message:\n', message)
        begin = time.time()
        stream = self.client.chat.completions.create(
            model=self.model,  # Model name to use
            messages=history_openai_format,  # Chat history
            temperature=self.temperature,  # Temperature for text generation
            top_p = self.top_p,
            stream=True,  # Stream response
            extra_body={
                'repetition_penalty':
                1,
                'stop': ["<|im_end|>"],
                'stop_token_ids': [
                    int(id.strip()) for id in self.stop_token_ids.split(',')
                    if id.strip()
                ] if self.stop_token_ids else []
        })

        end = time.time()
        print('consume_time:', round(end - begin))
        torch.cuda.empty_cache()

        response = ""
        for chunk in stream:
            response += (chunk.choices[0].delta.content or "")

        print('response:', response)
        # diagnosis = sample["入院诊断"]
        save_key = "诊断依据" if self.inf_task == "reasoning" else "治疗计划"
        reason_dict = self.get_result_dict(sample, response, dia4reasoning, response_key=save_key)
        return reason_dict

    def wrap_input_message(self, sample):
        inference_block_str, inference_str = "", ""
        dia_split = sample["入院诊断"].split(";")
        dia4reasoning = [diag for diag in dia_split if diag in self.disease_list]
        dia4reasoning = ";".join(dia4reasoning)
        main_diagnosis = dia_split[0]
        info_key = "诊断" if self.inf_task == "reasoning" else "治疗"
        if self.with_inference_ref:
            inference_info = self.get_inference(main_diagnosis)
            if inference_info:
                inference_str = self.inference_dict2str(inference_info[info_key])
            else:
                inference_str = f"{main_diagnosis}无参考信息"
            # inference_str = self.inference_dict2str(inference_info) if inference_info else f"{main_diagnosis}无参考信息"
            inference_block_str = self.block_flag
        
        shot_block_str= ""
        if self.with_shot:
            shot_block_str = self.block_flag

        message = shot_block_str + self.shot_ind + self.shot_str + \
                    self.block_flag + self.admission_ind + sample['symptom'] + \
                    self.block_flag + self.diagnosis_ind + dia4reasoning + \
                    inference_block_str + self.inference_ind + inference_str + self.block_flag + \
                    self.block_flag + self.prompt_str 
        return message, dia4reasoning
                
    def run(self):
        for classid, sample_list in self.sample_data.items(): # for sample in sample_data:
            self.out_dict[classid] = []
            for sample in sample_list:
                self.case_count += 1
                self.save_json(self.out_dict)
                reason_dict = self.get_llm_response(sample)
                self.out_dict[classid].append(reason_dict)

        self.save_json(self.out_dict)

if __name__ == "__main__":
    inference_api = InferenceAPI('config/inference_api.json')
    # treatment_api = TreatmentAPI('config/treatment_debug.json')
    inference_api.run()
