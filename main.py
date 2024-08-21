import json
from huggingface_hub import login
import os


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


def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output,model_file):
    if 'llama-3' in model_file.lower():

        if model_output:
            con = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant","content":model_output}
            ]
            return tokenizer.apply_chat_template(con)[:-5]
        else:
            con = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ]
            return tokenizer.apply_chat_template(con)
    else:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
        dialog_tokens = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"

        return tokenizer(dialog_tokens).input_ids


template = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
"""

template2 = """Given a statement of you: "You {}."
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
from PAlign.llama_pas import get_model
from copy import deepcopy

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
global_result = {}
global_cnt = {}


def getItems(filename):
    with open(filename+'/selected_IPIP300_samples.json','r',encoding='utf-8') as f:
        data = json.load(f)
    with open(filename+'/mpi_300_split.json',encoding='utf-8') as f:
        split_data = json.load(f)

    return data,pd.read_excel(filename+'/IPIP-NEO-ItemKey.xls'),split_data['train_index'],split_data['test_index']


def generateAnswer(tokenizer, model, dataset, template, scores=SCORES,system_prompt=system_prompt,model_file=None):
    if '70B' in model_file:
        batch_size = 3
    else:
        batch_size = 10
    questions, answers = [], []
    for item in dataset:
        questions.append(item["text"].lower())

    for batch in range(0, len(questions), batch_size):
        with torch.no_grad():
            outputs = model.generate(
                [prompt_to_tokens(tokenizer, system_prompt, template.format(prompt), 'Option',model_file) for prompt in
                 questions[batch:batch + batch_size]],
                max_new_tokens=15,
                # top_k=0,
            )

            output_text = tokenizer.batch_decode(outputs)
            if 'llama-3' in model_file.lower():
                answer = [text.split("<|end_header_id|>")[3] for text in output_text]
            else:
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

def from_index_to_data(train_index,test_index,text_file,dataset,dataset_set):
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

def get_activate_layer(layer_num,activate_name):
    lower_bound = layer_num // 4
    upper_bound = layer_num - lower_bound
    step = (upper_bound - lower_bound + 1) // 5

    if 'A' in activate_name:
        value = lower_bound
    elif 'C' in activate_name:
        value = lower_bound + 1 * step
    elif 'E' in activate_name:
        value = lower_bound + 2 * step
    elif 'N' in activate_name:
        value = lower_bound + 3 * step
    elif 'O' in activate_name:
        value = lower_bound + 4 * step
    return value

def lmean(l):
    return sum(l)/len(l)

def main(mode=None,model_file='',model=None,tokenizer=None,dataset_set='OOD'):
    dataset,text_file,train_index,test_index = getItems('IPIP')

    print("-" * 40)
    print(f"Current Prompt: {template}")
    results = []

    data = from_index_to_data(train_index,test_index,text_file,dataset,dataset_set)

    if '70' in model_file and 'ITI' in mode:
        index = range(0,300,10)
        datas = []
        for i in index:
            datas.append(data[i])
        data = datas
    if mode == 'NO_CHANGE':
        model.reset_all()
        answers = generateAnswer(tokenizer, model, data[0]['test'], template,model_file=model_file)

        for i in data:
            global_result = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
            global_result_abs = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            answer_number = 0
            for item in i['test']:
                label = item["label_ocean"]
                key = item["key"]
                parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
                if parsed_result:
                    parsed_result = parsed_result.group()[0].upper()

                    score = abs(SCORES[parsed_result] - item['value'])
                    if label not in global_result_abs:
                        global_result_abs[label] = []
                    if label not in global_result:
                        global_result[label] = []

                    global_cnt[parsed_result] += 1
                    global_result_abs[label].append(score)
                    score = SCORES[parsed_result]
                    if key == 1:
                        global_result[label].append(score)
                    else:
                        global_result[label].append(6 - score)
                else:
                    global_cnt["UNK"] += 1
                answer_number += 1

            result = global_result
            result_abs = global_result_abs
            count = global_cnt

            mean_var = calc_mean_and_var(result)
            mean_var_abs = calc_mean_and_var(result_abs)

            result_file = {'case': i['test'][0]['case'], 'result': result, 'count': count,
                           'mean_ver': mean_var, 'mean_ver_abs': mean_var_abs}
            results.append(result_file)

    if mode == 'DPO':
        model.reset_all()
        num = 0
        continue_sign = True
        for i in tqdm(data):
            if '3' in model_file:
                if f'final_checkpoint_model_{num}' in os.listdir('../llama3'):
                    continue_sign = False
                    adapter = './llama3'+f'/final_checkpoint_model_{num}'
            else:
                if f'final_checkpoint_model_{num}' in os.listdir('./llama2'):
                    continue_sign = False
                    adapter = '../llama2' + f'/final_checkpoint_model_{num}'
            num += 1
            if continue_sign:
                continue
            model, tokenizer = get_model(model_file,adapter=adapter)
            answers = generateAnswer(tokenizer, model, data[0]['test'], template,scores=SCORES,model_file=model_file)
            global_result = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
            global_result_abs = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            answer_number = 0
            for item in i['test']:
                label = item["label_ocean"]
                key = item["key"]
                parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
                if parsed_result:
                    parsed_result = parsed_result.group()[0].upper()

                    score = abs(SCORES[parsed_result] - item['value'])
                    if label not in global_result_abs:
                        global_result_abs[label] = []
                    if label not in global_result:
                        global_result[label] = []

                    global_cnt[parsed_result] += 1
                    global_result_abs[label].append(score)
                    score = SCORES[parsed_result]
                    if key == 1:
                        global_result[label].append(score)
                    else:
                        global_result[label].append(6 - score)
                else:
                    global_cnt["UNK"] += 1
                answer_number += 1

            result = global_result
            result_abs = global_result_abs
            count = global_cnt

            mean_var = calc_mean_and_var(result)
            mean_var_abs = calc_mean_and_var(result_abs)

            result_file = {'case': i['test'][0]['case'], 'result': result, 'count': count,
                           'mean_ver': mean_var, 'mean_ver_abs': mean_var_abs}
            results.append(result_file)


    if mode == 'few-shot':
        model.reset_all()

        for i in tqdm(data):
            system_prompt_text = 'Here are some of your behaviors and your level of recognition towards them' + \
                                 ';'.join([it['text'] + ':' + SCORES_BACK[it['value']] for it in i['train']])
            answers = generateAnswer(tokenizer, model, data[0]['test'], template,scores=SCORES,system_prompt=system_prompt_text,model_file=model_file)
            global_result = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
            global_result_abs = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            answer_number = 0
            for item in i['test']:
                label = item["label_ocean"]
                key = item["key"]
                parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
                if parsed_result:
                    parsed_result = parsed_result.group()[0].upper()

                    score = abs(SCORES[parsed_result] - item['value'])
                    if label not in global_result_abs:
                        global_result_abs[label] = []
                    if label not in global_result:
                        global_result[label] = []

                    global_cnt[parsed_result] += 1
                    global_result_abs[label].append(score)
                    score = SCORES[parsed_result]
                    if key == 1:
                        global_result[label].append(score)
                    else:
                        global_result[label].append(6 - score)
                else:
                    global_cnt["UNK"] += 1
                answer_number += 1

            result = global_result
            result_abs = global_result_abs
            count = global_cnt

            mean_var = calc_mean_and_var(result)
            mean_var_abs = calc_mean_and_var(result_abs)

            result_file = {'case': i['test'][0]['case'], 'result': result, 'count': count,
                           'mean_ver': mean_var, 'mean_ver_abs': mean_var_abs}
            results.append(result_file)

    if mode == 'personality_prompt':
        model.reset_all()

        system_prompt = json.load(open('PAPI/personality_prompt'))
        print(system_prompt[0]['output'][0])
        global_result = {}
        global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}

        for index,i in enumerate(tqdm(data)):
            system_prompt_text = system_prompt[index]['output'][0]
            answers = generateAnswer(tokenizer, model, data[0]['test'], template,system_prompt=system_prompt_text,model_file=model_file)
            global_result = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
            global_result_abs = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            answer_number = 0
            for item in i['test']:
                label = item["label_ocean"]
                key = item["key"]
                parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
                if parsed_result:
                    parsed_result = parsed_result.group()[0].upper()

                    score = abs(SCORES[parsed_result] - item['value'])
                    if label not in global_result_abs:
                        global_result_abs[label] = []
                    if label not in global_result:
                        global_result[label] = []

                    global_cnt[parsed_result] += 1
                    global_result_abs[label].append(score)
                    score = SCORES[parsed_result]
                    if key == 1:
                        global_result[label].append(score)
                    else:
                        global_result[label].append(6 - score)
                else:
                    global_cnt["UNK"] += 1
                answer_number += 1

            result = global_result
            result_abs = global_result_abs
            count = global_cnt

            mean_var = calc_mean_and_var(result)
            mean_var_abs = calc_mean_and_var(result_abs)

            result_file = {'case': i['test'][0]['case'], 'result': result, 'count': count,
                           'mean_ver': mean_var, 'mean_ver_abs': mean_var_abs}
            results.append(result_file)

    if mode == 'PAS':
        presonal_data = []
        for presonal in ['A', 'C', 'E', 'N', 'O']:
            for it in data[0]['train']:
                if it['label_ocean'] == presonal:
                    presonal_data.append(
                        {'question': template2.format(it['text']), 'answer_matching_behavior': 'A',
                         'answer_not_matching_behavior': 'E'})

        all_head_wise_activations = model.preprocess_activate_dataset(presonal_data)
        for index,i in enumerate(tqdm(data)):
            model.reset_all()
            system_prompt_text = 'Here are some of your behaviors and your level of recognition towards them' + \
                                 ';'.join([it['text'] + ':' + SCORES_BACK[it['value']] for it in i['train']])
            presonal_activate = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
            labels = []
            head_wise_activations = []

            presonal_number = 0
            for presonal in ['A','C','E','N','O']:
                for it in i['train']:
                    if it['label_ocean'] == presonal:
                        if it['value'] != 3 and it['value'] != 0:
                            if it['value'] > 3:
                                labels.append(1)
                                labels.append(0)
                                head_wise_activations.append(deepcopy(all_head_wise_activations[presonal_number]))
                                head_wise_activations.append(deepcopy(all_head_wise_activations[presonal_number+1]))
                            else:
                                labels.append(0)
                                labels.append(1)
                                head_wise_activations.append(deepcopy(all_head_wise_activations[presonal_number]))
                                head_wise_activations.append(deepcopy(all_head_wise_activations[presonal_number+1]))
                        presonal_number += 2

            activate = model.get_activations(deepcopy(all_head_wise_activations),labels,num_to_intervene=24)
            presonal_activate[presonal] = activate

            result_cache = []
            for num in [0,1,2,4,6,8]:
                model.reset_all()
                model.set_activate(activate, num)
                answers = generateAnswer(tokenizer, model, data[0]['test'], template, system_prompt=system_prompt_text,
                                         model_file=model_file)
                global_result = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
                global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
                global_result_abs = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
                answer_number = 0
                for item in i['test']:
                    label = item["label_ocean"]
                    key = item["key"]
                    parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
                    if parsed_result:
                        parsed_result = parsed_result.group()[0].upper()

                        score = abs(SCORES[parsed_result] - item['value'])
                        if label not in global_result_abs:
                            global_result_abs[label] = []
                        if label not in global_result:
                            global_result[label] = []

                        global_cnt[parsed_result] += 1
                        global_result_abs[label].append(score)
                        score = SCORES[parsed_result]
                        if key == 1:
                            global_result[label].append(score)
                        else:
                            global_result[label].append(6 - score)
                    else:
                        global_cnt["UNK"] += 1
                    answer_number += 1

                result = global_result
                result_abs = global_result_abs
                count = global_cnt

                mean_var = calc_mean_and_var(result)
                mean_var_abs = calc_mean_and_var(result_abs)


                result_file = {'case': i['test'][0]['case'], 'result': result, 'result_abs':result_abs,'count': count,
                               'mean_ver': mean_var,'mean_ver_abs':mean_var_abs}
                result_cache.append(result_file)

            scores = []
            for p in result_cache:
                score = sum([k[1] for k in p['mean_ver_abs']['mean']])
                if str(score) == 'nan':
                    score = 1e6
                scores.append(score)
            rs = result_cache[np.array(scores).argmin()]
            rs['alpha'] = result_cache[np.array(scores).argmin()]
            results.append(rs)

    print('*******Finally:******')
    mean = [i['mean_ver']['mean'] for i in results]
    mean_A = [i[0][1] for i in mean]
    mean_C = [i[1][1] for i in mean]
    mean_E = [i[2][1] for i in mean]
    mean_N = [i[3][1] for i in mean]
    mean_O = [i[4][1] for i in mean]


    std = [i['mean_ver']['std'] for i in results]
    std_A = [i[0][1] for i in std]
    std_C = [i[1][1] for i in std]
    std_E = [i[2][1] for i in std]
    std_N = [i[3][1] for i in std]
    std_O = [i[4][1] for i in std]

    mean_abs = [i['mean_ver_abs']['mean'] for i in results]
    mean_A_abs = [i[0][1] for i in mean_abs]
    mean_C_abs = [i[1][1] for i in mean_abs]
    mean_E_abs = [i[2][1] for i in mean_abs]
    mean_N_abs = [i[3][1] for i in mean_abs]
    mean_O_abs = [i[4][1] for i in mean_abs]


    std_abs = [i['mean_ver_abs']['std'] for i in results]
    std_A_abs = [i[0][1] for i in std_abs]
    std_C_abs = [i[1][1] for i in std_abs]
    std_E_abs = [i[2][1] for i in std_abs]
    std_N_abs = [i[3][1] for i in std_abs]
    std_O_abs = [i[4][1] for i in std_abs]


    with open('./log/{}_{}_{}.json'.format(mode,model_file.split('/')[-1],dataset_set),'w',encoding='utf-8') as f:
        log = {'score':{'mean_A':lmean(mean_A),'mean_C':lmean(mean_C),'mean_E':lmean(mean_E),'mean_N':lmean(mean_N),'mean_O':lmean(mean_O),
                        'std_A':lmean(std_A),'std_C':lmean(std_C),'std_E':lmean(std_E),'std_N':lmean(std_N),'std_O':lmean(std_O),
                        'mean_A_abs': lmean(mean_A_abs), 'mean_C_abs': lmean(mean_C_abs), 'mean_E_abs': lmean(mean_E_abs),
                        'mean_N_abs': lmean(mean_N_abs), 'mean_O_abs': lmean(mean_O_abs),
                        'std_A_abs': lmean(std_A_abs), 'std_C_abs': lmean(std_C_abs), 'std_E_abs': lmean(std_E_abs), 'std_N_abs': lmean(std_N_abs),
                        'std_O_abs': lmean(std_O_abs)
                        }}

        pprint(log)
        log['mean']={'A':mean_A,'C':mean_C,'E':mean_E,'N':mean_N,'O':mean_O}
        log['std']={'A':std_A,'C':std_C,'E':std_E,'N':std_N,'O':std_O}
        log['results'] = results
        json.dump(log,f,ensure_ascii=False,indent=4)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="mode")
    parser.add_argument("--modes", default='PAS',help="Name of the user to greet")
    parser.add_argument("--model_file", default='/zhuminjun/model/Meta-Llama-3-8B-Instruct-hf',help="Name of the user to greet")
    args = parser.parse_args()

    model_file = args.model_file
    modes = args.modes

    model, tokenizer = get_model(model_file)
    if 'llama-3' in model_file.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    for mode in modes:
        main(mode=mode,model_file=model_file,model=model,tokenizer=tokenizer,dataset_set='OOD')
