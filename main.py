import json
import logging
import pickle
from huggingface_hub import login
import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pprint import pprint
from PAlign.pas import get_model
from copy import deepcopy
from baseline_utils import process_answers, process_few_shot, calc_mean_and_var, process_personality_prompt

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
global_result = {}
global_cnt = {}

# Define constants
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

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

# Templates for personality assessment
TEMPLATE = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
"""


def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output, model_file):
    """
    Convert prompts to tokens based on the model type.
    """
    if 'llama-3' in model_file.lower():
        if model_output:
            con = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": model_output}
            ]
            return tokenizer.apply_chat_template(con)[:-1]
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


def getItems(filename):
    """
    Load data from files.

    Test-set.json: 300 samples for Test, if you need to load more, you can download from huggingface (https://huggingface.co/datasets/WestlakeNLP/PAPI-300K)
    mpi_300_split.json: The original IPIP-NEO-300 questionnaire includes the IPIP-NEO-120 and the remaining 180 questions.
        Existing psychological research indicates that the IPIP-NEO-120 is sufficient to represent a person's personality traits.
        Therefore, we first use the IPIP-NEO-120 to concretize a person's characteristics,
        and then divide the remaining 180 questions into a section that we ask the model to predict.
    IPIP-NEO-ItemKey.xls:  This is where we store the question text corresponding to each question ID (the original data only contains the ID part).
    """
    with open(filename + '/Test-set.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(filename + '/mpi_300_split.json', encoding='utf-8') as f:
        split_data = json.load(f)
    return data, pd.read_excel(filename + '/IPIP-NEO-ItemKey.xls'), split_data['train_index'], split_data['test_index']


def generateAnswer(tokenizer, model, dataset, template, scores=SCORES, system_prompt=SYSTEM_PROMPT, model_file=None, raw_logger=None):
    """
    Generate answers using the model.
    """
    batch_size = 3
    questions = [item["text"].lower() for item in dataset]
    answers = []

    for batch in range(0, len(questions), batch_size):
        with torch.no_grad():
            outputs = model.generate(
                [prompt_to_tokens(tokenizer, system_prompt, template.format(prompt), 'Option', model_file)
                 for prompt in questions[batch:batch + batch_size]],
                max_new_tokens=15,
            )
            output_text = tokenizer.batch_decode(outputs)
            if 'llama-3' in model_file.lower():
                answer = [text.split("<|end_header_id|>")[-1] for text in output_text]
            else:
                answer = [text.split("[/INST]")[-1] for text in output_text]
            if raw_logger:
                for i, ans in enumerate(answer):
                    raw_logger.info(f"q={len(answers) + i} | {ans.strip()}")
            answers.extend(answer)

    return answers


def calc_mean_and_var(result):
    """
    Calculate mean and variance of results.
    """
    mean = {key: np.mean(np.array(item)) for key, item in result.items()}
    std = {key: np.std(np.array(item)) for key, item in result.items()}
    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }


def from_index_to_data(train_index, test_index, text_file, dataset, dataset_set):
    """
    Convert indexed data to structured format.
    """
    data = []
    for i in tqdm(dataset):
        d_train = []
        d_test = []
        for t_i in train_index:
            t = text_file[text_file['Full#'] == t_i].iloc[0].to_list()
            item = {
                'label_raw': t[4],
                'text': t[5],
                'label_ocean': t[3][0],
                'key': {'+': 1, '-': -1}[t[2][0]]
            }
            exec(f"item['value'] = i['i{t_i}']")
            item['case'] = i['case']
            d_train.append(item)
        for t_i in test_index:
            t = text_file[text_file['Full#'] == t_i].iloc[0].to_list()
            item = {
                'label_raw': t[4],
                'text': t[5],
                'label_ocean': t[3][0],
                'key': {'+': 1, '-': -1}[t[2][0]]
            }
            exec(f"item['value'] = i['i{t_i}']")
            item['case'] = i['case']
            d_test.append(item)
        data.append({'train': d_train, 'test': d_test})
    return data


def get_activate_layer(layer_num, activate_name):
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
    """
    Calculate mean of a list.
    """
    return sum(l) / len(l)


def setup_raw_logger(output_dir='./reproduction'):
    """Set up a dedicated file logger for raw model generations."""
    os.makedirs(output_dir, exist_ok=True)
    raw_logger = logging.getLogger('raw_generations')
    raw_logger.setLevel(logging.INFO)
    if not raw_logger.handlers:
        fh = logging.FileHandler(os.path.join(output_dir, 'raw_generations.log'), mode='a')
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%dT%H:%M:%S'))
        raw_logger.addHandler(fh)
    return raw_logger


def process_pas(data, model, tokenizer, model_file, output_dir='./reproduction'):
    """
    Process data using Persona Activation Steering (PAS) method.

    :param data: List of personality data samples
    :param model: The language model to use
    :param tokenizer: The tokenizer for the model
    :param model_file: Path to the model file
    :param output_dir: Directory for output files
    :return: List of results for each personality sample
    """
    raw_logger = setup_raw_logger(output_dir)

    # Set up resume directories
    os.makedirs(os.path.join(output_dir, 'subject_results'), exist_ok=True)
    progress_path = os.path.join(output_dir, 'pas_progress.jsonl')

    # Load existing results for resume
    results = [None] * len(data)
    done_indices = set()
    for idx in range(len(data)):
        pkl_path = os.path.join(output_dir, 'subject_results', f'subject_{idx:04d}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                results[idx] = pickle.load(f)
            done_indices.add(idx)
    if done_indices:
        print(f"Resuming: {len(done_indices)} subjects already completed, skipping them.")

    # Prepare personal data for activation
    personal_data = []
    for personal in ['A', 'C', 'E', 'N', 'O']:
        for item in data[0]['train']:
            if item['label_ocean'] == personal:
                personal_data.append({
                    'question': TEMPLATE.format(item['text']),
                    'answer_matching_behavior': 'A',
                    'answer_not_matching_behavior': 'E'
                })

    # Preprocess activation dataset
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)

    for index, sample in enumerate(tqdm(data)):
        if index in done_indices:
            continue

        model.reset_all()

        # Generate system prompt from training data
        system_prompt_text = 'Here are some of your behaviors and your level of recognition towards them;' + \
                             ';'.join([f"{it['text']}:{SCORES_BACK[it['value']]}" for it in sample['train']])

        # Prepare labels and activations for each personality trait
        labels = []
        head_wise_activations = []
        personal_number = 0
        for personal in ['A', 'C', 'E', 'N', 'O']:
            for item in sample['train']:
                if item['label_ocean'] == personal:
                    if item['value'] not in [0, 3]:
                        if item['value'] > 3:
                            labels.extend([1, 0])
                        else:
                            labels.extend([0, 1])
                        head_wise_activations.extend([
                            deepcopy(all_head_wise_activations[personal_number]),
                            deepcopy(all_head_wise_activations[personal_number + 1])
                        ])
                    personal_number += 2

        # Get activations for intervention
        activate = model.get_activations(deepcopy(head_wise_activations), labels, num_to_intervene=24)

        # Test different activation levels
        alpha_values = [0, 1, 2, 4, 6, 8]
        result_cache = []
        case_id = sample['test'][0]['case']
        for num in alpha_values:
            model.reset_all()
            model.set_activate(activate, num)

            raw_logger.info(f"=== subject={index} case={case_id} alpha={num} ===")

            answers = generateAnswer(tokenizer, model, data[0]['test'], TEMPLATE,
                                     system_prompt=system_prompt_text, model_file=model_file,
                                     raw_logger=raw_logger)

            # Process answers and calculate results
            result = process_answers(answers, sample)
            result_cache.append(result)

        # Select the best result based on mean absolute error
        scores = []
        for p in result_cache:
            score = sum([k[1] for k in p['mean_ver_abs']['mean']])
            if str(score) == 'nan':
                score = 1e6
            scores.append(score)
        best_idx = int(np.array(scores).argmin())
        rs = result_cache[best_idx]
        rs['alpha'] = alpha_values[best_idx]
        results[index] = rs

        # Save per-subject pickle for resume
        pkl_path = os.path.join(output_dir, 'subject_results', f'subject_{index:04d}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(rs, f)

        # Append progress line
        progress_entry = {
            'index': index,
            'case': case_id,
            'alpha': alpha_values[best_idx],
            'score_sum': scores[best_idx],
            'mean_abs': {k: v for k, v in rs['mean_ver_abs']['mean']},
            'timestamp': datetime.now().isoformat()
        }
        with open(progress_path, 'a') as f:
            f.write(json.dumps(progress_entry) + '\n')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Subject {index}/{len(data)} done | "
              f"case={case_id} | best_alpha={alpha_values[best_idx]} | "
              f"score_sum={scores[best_idx]:.3f}")

    # Filter out any None entries (shouldn't happen if all completed)
    results = [r for r in results if r is not None]
    return results


def print_and_save_results(results, mode, model_file, dataset_set, output_dir='./reproduction'):
    """
    Print and save the final results of the personality assessment.

    Args:
    results (list): List of result dictionaries from the assessment
    mode (str): The mode of assessment used
    model_file (str): The name of the model file used
    dataset_set (str): The dataset set used (e.g., 'OOD')
    """
    print('*******Finally:******')

    # Extract mean and standard deviation values
    mean = [i['mean_ver']['mean'] for i in results]
    mean_A, mean_C, mean_E, mean_N, mean_O = ([i[j][1] for i in mean] for j in range(5))

    std = [i['mean_ver']['std'] for i in results]
    std_A, std_C, std_E, std_N, std_O = ([i[j][1] for i in std] for j in range(5))

    mean_abs = [i['mean_ver_abs']['mean'] for i in results]
    mean_A_abs, mean_C_abs, mean_E_abs, mean_N_abs, mean_O_abs = ([i[j][1] for i in mean_abs] for j in range(5))

    std_abs = [i['mean_ver_abs']['std'] for i in results]
    std_A_abs, std_C_abs, std_E_abs, std_N_abs, std_O_abs = ([i[j][1] for i in std_abs] for j in range(5))

    # Prepare log dictionary
    log = {
        'score': {
            'mean_A': lmean(mean_A), 'mean_C': lmean(mean_C), 'mean_E': lmean(mean_E),
            'mean_N': lmean(mean_N), 'mean_O': lmean(mean_O),
            'std_A': lmean(std_A), 'std_C': lmean(std_C), 'std_E': lmean(std_E),
            'std_N': lmean(std_N), 'std_O': lmean(std_O),
            'mean_A_abs': lmean(mean_A_abs), 'mean_C_abs': lmean(mean_C_abs),
            'mean_E_abs': lmean(mean_E_abs), 'mean_N_abs': lmean(mean_N_abs),
            'mean_O_abs': lmean(mean_O_abs),
            'std_A_abs': lmean(std_A_abs), 'std_C_abs': lmean(std_C_abs),
            'std_E_abs': lmean(std_E_abs), 'std_N_abs': lmean(std_N_abs),
            'std_O_abs': lmean(std_O_abs)
        }
    }

    # Print the log
    pprint(log)

    # Add more details to the log
    log['mean'] = {'A': mean_A, 'C': mean_C, 'E': mean_E, 'N': mean_N, 'O': mean_O}
    log['std'] = {'A': std_A, 'C': std_C, 'E': std_E, 'N': std_N, 'O': std_O}

    # Save the log to a file
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f'{mode}_{model_file.split("/")[-1]}_{dataset_set}.json')
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {log_filename}")


def main(mode=None, model_file='', model=None, tokenizer=None, dataset_set='OOD', num_subjects=0, output_dir='./reproduction'):
    """
    Main function to run personality assessment.
    """
    dataset, text_file, train_index, test_index = getItems('PAPI')
    print("-" * 40)
    print(f"Current Prompt: {TEMPLATE}")
    results = []
    data = from_index_to_data(train_index, test_index, text_file, dataset, dataset_set)

    if num_subjects > 0:
        data = data[:num_subjects]
        print(f"Using {num_subjects} subjects (out of {len(dataset)} total)")

    if mode == 'NO_CHANGE':
        # Process data without any changes
        model.reset_all()
        answers = generateAnswer(tokenizer, model, data[0]['test'], TEMPLATE, model_file=model_file)
        results = process_answers(data, answers)

    elif mode == 'few-shot':
        # Process data using few-shot learning
        results = process_few_shot(data, model, tokenizer, model_file)

    elif mode == 'personality_prompt':
        # Process data using personality prompts
        results = process_personality_prompt(data, model, tokenizer, model_file)

    elif mode == 'PAS':
        # Process data using PAS (Personality Assessment System)
        results = process_pas(data, model, tokenizer, model_file, output_dir=output_dir)

    # Print and save final results
    print_and_save_results(results, mode, model_file, dataset_set, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Personality Alignment of LLMs")
    parser.add_argument("--modes", default='PAS', help="Assessment mode (PAS, NO_CHANGE, few-shot, personality_prompt)")
    parser.add_argument("--model_file", default='meta-llama/Meta-Llama-3-8B-Instruct', help="HuggingFace model name")
    parser.add_argument("--num_subjects", type=int, default=0, help="Number of subjects to process (0=all)")
    parser.add_argument("--output_dir", default='./reproduction', help="Output directory for results")
    args = parser.parse_args()

    model_file = args.model_file
    modes = [args.modes]
    num_subjects = args.num_subjects
    output_dir = args.output_dir

    model, tokenizer = get_model(model_file)
    if 'llama-3' in model_file.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    for mode in modes:
        main(mode=mode, model_file=model_file, model=model, tokenizer=tokenizer,
             dataset_set='OOD', num_subjects=num_subjects, output_dir=output_dir)
