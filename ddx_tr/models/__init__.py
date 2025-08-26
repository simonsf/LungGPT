import os

from utils.vllm_wrapper import vLLMWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_weight_dict = {
    "qwen-int4": "/data/MD_GPT/model_base/source_models/Qwen/Qwen-72B-Chat-Int4/",
    "qwen-int8": "/data5/Qwen/Qwen-72B-Chat-Int8/",
}


def init_model(args):
    """
    initialize model refer to args
    :param args: moc
    :return: model tokenizer config
    """
    print("init model ...")
    checkpoint_path = model_weight_dict[args.model]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto", trust_remote_code=True).eval()
    config = GenerationConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    config.max_new_tokens = args.max_tokens
    config.temperature = args.temperature
    config.top_p = args.top_p
    print(config)

    return model, tokenizer, config


def init_vllm_model(args):
    print("init model ...")
    num_gpus = args.num_gpus
    checkpoint_path = model_weight_dict[args.model]
    vllm_model = vLLMWrapper(checkpoint_path,
                             quantization='gptq',
                             dtype="float16",
                             tensor_parallel_size=num_gpus)
    config = GenerationConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    config.max_new_tokens = 32000
    config.temperature = 0.9
    config.max_window_size = 10240
    config.top_p = 0.7
    print(config)

    return vllm_model, config
