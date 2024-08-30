import re
import numpy as np
from tqdm import tqdm

# Constants (you might want to import these from a common config file)
SCORES = {
    "A": 5, "B": 4, "C": 3, "D": 2, "E": 1
}

SCORES_BACK = {
    5: 'Very Accurate',
    4: 'Moderately Accurate',
    3: 'Neither Accurate Nor Inaccurate',
    2: 'Moderately Inaccurate',
    1: 'Very Inaccurate',
    0: 'Unknown'
}


def calc_mean_and_var(result):
    """
    Calculate mean and variance of results.
    """
    mean = {}
    std = {}
    for key, item in result.items():
        mean[key] = np.mean(np.array(item))
        std[key] = np.std(np.array(item))

    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }



def process_answers(answers,sample):
    """
    Process answers and calculate results.
    """

    global_result = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
    global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
    global_result_abs = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
    answer_number = 0
    for item in sample['test']:
        label = item["label_ocean"]
        key = item["key"]
        parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
        if parsed_result:
            parsed_result = parsed_result.group()[0].upper()
            score = abs(SCORES[parsed_result] - item['value'])
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

    mean_var = calc_mean_and_var(global_result)
    mean_var_abs = calc_mean_and_var(global_result_abs)
    result_file = {
        'case': sample['test'][0]['case'],
        'result': global_result,
        'count': global_cnt,
        'mean_ver': mean_var,
        'mean_ver_abs': mean_var_abs
    }

    return result_file


def process_few_shot(data, model, tokenizer, model_file, template):
    """
    Process data using few-shot learning method.
    """
    results = []
    for i in tqdm(data):
        system_prompt_text = 'Here are some of your behaviors and your level of recognition towards them' + \
                             ';'.join([f"{it['text']}:{SCORES_BACK[it['value']]}" for it in i['train']])
        answers = generate_answer(tokenizer, model, data[0]['test'], template, scores=SCORES,
                                  system_prompt=system_prompt_text, model_file=model_file)
        results.extend(process_answers([i], answers))
    return results


def process_personality_prompt(data, model, tokenizer, model_file):
    """
    Process data using personality prompts method.
    """
    system_prompt = json.load(open('PAPI/personality_prompt'))
    results = []
    for index, i in enumerate(tqdm(data)):
        system_prompt_text = system_prompt[index]['output'][0]
        answers = generateAnswer(tokenizer, model, data[0]['test'], TEMPLATE, system_prompt=system_prompt_text,
                                 model_file=model_file)
        results.extend(process_answers([i], answers))
    return results
