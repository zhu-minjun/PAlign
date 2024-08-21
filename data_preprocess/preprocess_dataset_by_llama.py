import json
from transformers import LlamaForCausalLM,AutoTokenizer
from transformers import AutoModelForCausalLM,BitsAndBytesConfig, pipeline
from tqdm import tqdm
SCORES = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
}
SCORES_BACK = {
    5: 'Very Accurate',
    4: 'Moderately Accurate',
    3: 'Neither Accurate Nor Inaccurate',
    2: 'Moderately Inaccurate',
    1: 'Very Inaccurate',
    0: 'Unknown'
}

system_prompt = "You are a helpful, honest and concise assistant."
import os
os.chdir('/zhuminjun/PAlign')

def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
    dialog_tokens = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"

    return dialog_tokens


template = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
"""

global_result = {}
global_cnt = {}

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm,trange
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pprint import pprint
from ControlLM.llama import get_model

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
global_result = {}
global_cnt = {}

def getItems(filename):
    with open(filename+'/selected_IPIP300_samples.json','r',encoding='utf-8') as f:
        data = json.load(f)
    with open(filename+'/mpi_300_split.json',encoding='utf-8') as f:
        split_data = json.load(f)

    return data,pd.read_excel(filename+'/IPIP-NEO-ItemKey.xls'),split_data['train_index'],split_data['test_index']


def generateAnswer(tokenizer, model, dataset, template, scores=SCORES,system_prompt=system_prompt):

    batch_size = 24
    questions, answers = [], []
    for _, item in dataset.iterrows():
        questions.append(item["text"].lower())

    for batch in range(0, len(questions), batch_size):
        outputs = model.generate(
            [prompt_to_tokens(tokenizer, system_prompt, template.format(prompt), 'Option') for prompt in
             questions[batch:batch + batch_size]],
            temperature=0.0,
            max_new_tokens=15,
            top_p=0.95,
            # top_k=0,
        )

        output_text = tokenizer.batch_decode(outputs)
        answer = [text.split("[/INST]")[-1] for text in output_text]
        answers.extend(answer)

    return answers


def calc_mean_and_var(result):
    mean = {}
    std = {}
    for key, item in result.items():
        mean[key] = np.mean(np.array(item))
        std[key] = np.std(np.array(item))

    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }

def from_index_to_data(train_index,test_index,text_file,dataset):
    data = []
    for i in tqdm(dataset):
        d_train = []
        d_test = []
        for t_i in train_index:
            t = text_file[text_file['Full#']==t_i].iloc[0].to_list()
            item = {'label_raw': t[4],
                     'text': t[5],
                     'label_ocean': t[3][0],
                     'key': {'+':1,'-':-1}[t[2][0]]}
            exec("""item['value'] = i['i{}']""".format(t_i))
            item['case'] = i['case']
            d_train.append(item)
        for t_i in test_index:
            t = text_file[text_file['Full#']==t_i].iloc[0].to_list()
            item = {'label_raw': t[4],
                     'text': t[5],
                     'label_ocean': t[3][0],
                     'key': {'+':1,'-':-1}[t[2][0]]}
            exec("""item['value'] = i['i{}']""".format(t_i))
            item['case'] = i['case']
            d_test.append(item)
        data.append({'train':d_train,'test':d_test})
    return data

def main():
    dataset,text_file,train_index,test_index = getItems('IPIP')

    data = from_index_to_data(train_index,test_index,text_file,dataset)

    result = []
    for i in tqdm(data):
        try:


            prompt_text = 'Please use "You" to organize and rewrite the following fragment into a smooth object description, without including any "accurate" and "inaccurate" elements, but rather a natural description:\n' + \
                                 ';'.join([it['text'] + ':' + SCORES_BACK[it['value']] for it in i['train']])
            messages = [[
                {"role": "system",
                 "content": "You are a helpful assistant."},
                {'role': 'user', 'content': prompt_text+'\n\nHere is a rewritten version of the text, organized into a smooth object description:'}]]
            output_by_gpt = get_llama(messages)
            print(output_by_gpt)
            result.append({'input':prompt_text,'output':output_by_gpt})
        except:
            print('ERRPR'+prompt_text)
    with open('IPIP/auto_prompt.json','w',encoding='utf-8') as f:
        json.dump(result,f,indent=4,ensure_ascii=False)


import requests
import json


def get_llama2(prompt_text):
    # URL of the chathf endpoint
    url = "http://127.0.0.1:37200/chathf"

    # Data to be sent in POST request
    data = {
        "dialogs": prompt_text,
        "temperature": 0.6,  # You can adjust this if needed
        "top_p": 0.9,  # You can adjust this if needed
        "max_gen_len": 1024  # Maximum number of tokens to generate
    }

    # Convert the dictionary to a JSON object
    headers = {'Content-Type': 'application/json'}

    # Sending POST request to the server
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Parsing the JSON response
        response_data = response.json()
        # Retrieve the results from the response
        results = response_data.get('results', [])
        return '\n'.join(results)
    else:
        # Print error if something goes wrong
        return "Error: " + response.text


def get_llama(prompt_text):


    # Data to be sent in POST request
    data = {
        "dialogs": prompt_text,
        "temperature": 0.6,  # You can adjust this if needed
        "top_p": 0.9,  # You can adjust this if needed
        "max_gen_len": 1024  # Maximum number of tokens to generate
    }

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    input_ids = tokenizer.apply_chat_template(data['dialogs'], padding=True, return_tensors='pt').to(0)
    results = model.generate(input_ids, max_new_tokens=data['max_gen_len'], temperature=data['temperature'], eos_token_id=terminators)
    outputs = [i.split('<|eot_id|>')[0] for i in tokenizer.batch_decode(results[:,len(input_ids[0]):])]

    return outputs



if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        '/zhuminjun/model/Meta-Llama-3-70B-Instruct-hf',
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,

    )
    tokenizer = AutoTokenizer.from_pretrained('Meta-Llama-3-70B-Instruct-hf')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    main()
