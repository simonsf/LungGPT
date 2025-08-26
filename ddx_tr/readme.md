# Code for diagnosis basis and treatment plan generating

> [!Important]
> We provide a more accurate augmented generating way based on well-structralized json dataset. If you are interested in the based RAG version, please referring to the [cookbook](https://huggingface.co/learn/cookbook/advanced_rag) in hugging face, that is what we use in paper. But we found that sometimes lungchain retrieved wrong documents, because the same diagnosis may appear in different documents in our database. There is only one documents with the right diagnosis and treatment information. We strongly suggest you to retrieve documents in key-value way of medical field.

## Requirements

* python 3.9 and above
* pytorch 2.1 and above
* transformers 4.32 and above

## Quickstart

Below, we provide simple examples to show how to use LLM 🤖 to generate diagnosis basis and treatment plan based on dmission record.

Please make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.

```bash
pip install -r requirements.txt
```

Then download the [qwen](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524) or other model in hugging face supported by vllm, start the server via the vllm server command or through [docker](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html).

```bash
vllm serve <path of model download> \
    --tensor-parallel-size 4 \
    --host 127.0.0.1 \
    --port 8001 \
    --served-model-name Qwen \
    --api-key xxx \
    --gpu-memory-utilization 0.95 \
    --max_model_len 4096 \
    --enforce-eager \
    --quantization gptq
```
`tensor-parallel-size` means number of GPUs to deployed the LLM 🤖, and you can modify the host and port as you want.

Modify JSON config file before running.
```
{
    "model_url": "http://127.0.0.1:8001/v1",  # the same as the host parameter of above
    "model": "Qwen",  # the same as served-model-name
    "openai_api_key": "EMPTY",
    "temperature": 0.75,
    "top_p": 0.65,
    "with_shot": true, # generating with shot 
    "with_inference_ref": true, # generating with retrieved reference information  
    "inference_task": "treatment", # [treatment | reasoning] treatment means generating treatment plan, reasoning means generating diagnosis basis
    "disease_class": "data/reference/disease_all_class.json",  # all diseases standardizes by ICD , you can enlarge it
    "reference": "data/reference/reference.json",  # key-value database for augmented generating.
    "bookclass": "data/reference/class_from_book.json", # diseases from the book we collect for data structurization 
    "stop_token_ids": "",
    "system_role": "data/reference/model_role.json",
    "addmission_ind": "当前入院录:\n",
    "diagnosis_ind": "诊断结果:\n",
    "inference_ind": "参考信息:\n",
    "shot_ind": "参考例子:\n",
    "block_flag": "\n---\n",
    "save_path": "data/result",
    "test_data": "data/test_set/debug_cased.json"
}
```
We provided two admission docs for debugging the code, please prepare your data referring to these two case in `.data/test_set/debug_cased.json`. 
Aditionally, we provided two categories of disease  in `.data/reference/reference.json`, you can prepared your data at json format as I do.

Having modified the aboved config json file, just run the code to generate diagnosis or treatment plan.

```bash
python inference.py
```