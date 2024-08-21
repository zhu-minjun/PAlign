"""
Model Setup and Inference Script for Meta-LLaMA

This script defines a class for loading and interacting with the Meta-LLaMA model,
including handling model activations and generating responses. The script includes
methods for setting up the model, processing input datasets, and performing inference
with various configurations.
"""

import json
import random
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from einops import rearrange
import pickle
from functools import partial
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.rich import tqdm
import matplotlib
from baukit import Trace, TraceDict
from modeling_llama import LlamaForCausalLM
from copy import deepcopy

def get_model(model_name='meta-llama/Llama-2-7b-chat-hf', control_activate_name=None, control_layer=None, gamma=None, adapter=None):
    """
    Loads and sets up the Meta-LLaMA model for inference and activation handling.

    Args:
    model_name (str): Name of the model to load.
    control_activate_name (str): Name of the control activation (optional).
    control_layer (str): Specific layer to control (optional).
    gamma (float): Gamma value for model control (optional).
    adapter (str): Path to the adapter model (optional).

    Returns:
    model: Configured LLaMA model.
    tokenizer: Corresponding tokenizer.
    """

    class AttnWrapper(nn.Module):
        """
        Wrapper for the attention mechanism to capture activations.
        """
        def __init__(self, attn):
            super().__init__()
            self.attn = attn
            self.activations = None

        def forward(self, *args, **kwargs):
            output = self.attn(*args, **kwargs)
            self.activations = output[0]
            return output

    class BlockOutputWrapper(nn.Module):
        """
        Wrapper for model block to handle activations and transformations.
        """
        def __init__(self, block, unembed_matrix, norm, tokenizer):
            super().__init__()
            self.block = block
            self.unembed_matrix = unembed_matrix
            self.norm = norm
            self.tokenizer = tokenizer

            self.block.self_attn = AttnWrapper(self.block.self_attn)
            self.post_attention_layernorm = self.block.post_attention_layernorm

            self.attn_out_unembedded = None
            self.intermediate_resid_unembedded = None
            self.mlp_out_unembedded = None
            self.block_out_unembedded = None

            self.activations = None
            self.add_activations = None
            self.after_position = None

            self.save_internal_decodings = False

            self.calc_dot_product_with = None
            self.dot_products = []

        def reset(self):
            self.add_activations = None
            self.activations = None
            self.block.self_attn.activations = None
            self.after_position = None
            self.calc_dot_product_with = None
            self.dot_products = []

    class LlamaLM:
        """
        Main class for loading, configuring, and interacting with the Meta-LLaMA model.
        """
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if adapter:
                if 'ppo' in adapter:
                    load_model_name = adapter
                    load_adapter = None
                else:
                    load_adapter = adapter
                    load_model_name = model_name
            else:
                load_model_name = model_name
                load_adapter = None
            self.model_file = load_model_name

            if '70B' in self.model_file:
                self._load_large_model(load_model_name)
            else:
                self._load_standard_model(load_model_name, load_adapter)

            self.bias_cache = []
            for i, layer in enumerate(self.model.model.layers):
                self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm, self.tokenizer)
                self.bias_cache.append(deepcopy(self.model.model.layers[i].block.self_attn.attn.o_proj.bias))

        def _load_large_model(self, model_name):
            """
            Loads the large variant of the Meta-LLaMA model (70B).
            """
            model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
            self.weight_cache = [deepcopy(layer.self_attn.o_proj.weight).cuda() for layer in model.model.layers]
            model = None
            torch.cuda.empty_cache()
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            self.model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, quantization_config=quantization_config, torch_dtype=torch.bfloat16)

        def _load_standard_model(self, model_name, adapter):
            """
            Loads the standard variant of the Meta-LLaMA model.
            """
            if 'ppo' in model_name:
                self.model = LlamaForCausalLM.from_pretrained(model_name, device_map='cuda')
            else:
                self.model = LlamaForCausalLM.from_pretrained(model_name).half().cuda()
            if adapter:
                self.model.load_adapter(adapter)
            self.model.eval()
            self.device = self.model.device

        def generate(self, model, text, max_length=512, max_new_tokens=None):
            """
            Generates responses based on the input text using the model.

            Args:
            model: The model to use for generation.
            text (str): Input text for the model.
            max_length (int): Maximum length of the generated text.
            max_new_tokens (int): Maximum number of new tokens to generate.

            Returns:
            tokens: Generated tokens.
            """
            tokenizer = model.tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            stop_id = tokenizer.sep_token_id
            pad_id = tokenizer.pad_token_id

            device = model.device
            input_ids = [t for t in text]
            min_prompt_len = min(len(t) for t in input_ids)
            max_prompt_len = max(len(t) for t in input_ids)

            if max_new_tokens:
                max_length = max_prompt_len + max_new_tokens
            tokens = torch.full((len(input_ids), max_length), pad_id, dtype=torch.long).to(device)
            for k, t in enumerate(input_ids):
                tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            prev_pos = 0
            cur_pos = min_prompt_len - 1
            input_text_mask = tokens != pad_id
            eos_reached = torch.tensor([False] * len(input_ids), device=device)
            past_key_values = None

            with torch.no_grad():
                for cur_pos_add in range(max_length):
                    cur_pos += 1
                    if prev_pos != 0:
                        prev_pos = cur_pos - 1
                    if tokens.shape[1] == cur_pos:
                        break
                    torch.cuda.empty_cache()

                    logits = model.model(tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=past_key_values)
                    next_token = torch.topk(logits['logits'][:, -1], 1, dim=-1)[1][:, -1]
                    next_token = next_token.reshape(-1)
                    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                    tokens[:, cur_pos] = next_token
                    eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == model.tokenizer.eos_token_id)

                    if all(eos_reached):
                        break
                    prev_pos = cur_pos
                    past_key_values = logits["past_key_values"]
            return tokens

        def __call__(self, input_ids):
            """
            Calls the model with the given input IDs.

            Args:
            input_ids: Input IDs for the model.

            Returns:
            logits: Model output logits.
            """
            with torch.no_grad():
                logits = self.model(input_ids)
                return logits

        def get_last_activations(self, layer):
            """
            Retrieves the last activations for a specific layer.

            Args:
            layer: Layer number.

            Returns:
            activations: Activations of the specified layer.
            """
            return self.model.model.layers[layer].activations

        def reset_all(self):
            """
            Resets all internal states and biases of the model.
            """
            for i, layer in enumerate(self.model.model.layers):
                layer.reset()
                self.model.model.layers[i].block.self_attn.attn.o_proj.bias = self.bias_cache[i]

        def get_activations(self, all_head_wise_activations, labels, num_to_intervene=48):
            def get_top_heads(separated_activations, separated_labels, num_layers, num_heads, num_to_intervene):

                probes, all_head_accs_np = train_probes(separated_activations,
                                                        separated_labels, num_layers=num_layers, num_heads=num_heads)
                all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads,2)
                all_head_accs_np = all_head_accs_np.mean(2)
                top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
                top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

                return top_heads, probes

            def train_probes(separated_head_wise_activations, separated_labels,
                             num_layers, num_heads):

                all_head_accs = []
                probes = []

                train_idxs = np.arange(len(separated_labels))

                # pick a val set using numpy
                train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * (1 - 0.4)),
                                                  replace=False)
                val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

                all_X_train = np.array([separated_head_wise_activations[i] for i in train_set_idxs])
                all_X_val = np.array([separated_head_wise_activations[i] for i in val_set_idxs])
                y_train = np.array([separated_labels[i] for i in train_set_idxs])
                y_val = np.array([separated_labels[i] for i in val_set_idxs])

                for layer in tqdm(range(num_layers)):
                    for head in range(num_heads):
                        X_train = all_X_train[:, layer, head, :]
                        X_val = all_X_val[:, layer, head, :]

                        clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
                        y_pred = clf.predict(X_train)
                        y_val_pred = clf.predict(X_val)
                        all_head_accs.append([accuracy_score(y_val, y_val_pred),accuracy_score(y_train,y_pred)])
                        probes.append(clf)

                all_head_accs_np = np.array(all_head_accs)

                return probes, all_head_accs_np

            def flattened_idx_to_layer_head(flattened_idx, num_heads):
                return flattened_idx // num_heads, flattened_idx % num_heads

            def layer_head_to_flattened_idx(layer, head, num_heads):
                return layer * num_heads + head

            def get_interventions_dict(top_heads, probes, tuning_activations, num_heads,com_directions=None):

                interventions = {}
                for layer, head in top_heads:
                    interventions[f"model.layers.{layer}.block.self_attn.attn.head_out"] = []
                for layer, head in top_heads:
                    if com_directions is not None:
                        direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
                    else:
                        direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
                    direction = direction / np.linalg.norm(direction)
                    activations = tuning_activations[:, layer, head, :]  # batch x 128
                    proj_vals = activations @ direction.T
                    proj_val_std = np.std(proj_vals)
                    interventions[f"model.layers.{layer}.block.self_attn.attn.head_out"].append(
                        (head, direction.squeeze(), proj_val_std))
                for layer, head in top_heads:
                    interventions[f"model.layers.{layer}.block.self_attn.attn.head_out"] = sorted(
                        interventions[f"model.layers.{layer}.block.self_attn.attn.head_out"], key=lambda x: x[0])

                return interventions

            def get_com_directions(num_layers, num_heads,usable_head_wise_activations,
                                   usable_labels):

                com_directions = []

                for layer in range(num_layers):
                    for head in range(num_heads):
                       # usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
                        #usable_head_wise_activations = np.concatenate(
                            #[separated_head_wise_activations[i][:, layer, head, :] for i in usable_idxs], axis=0)
                        #usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
                        usable_labels = np.array(usable_labels)
                        head_wise_activations = usable_head_wise_activations[:,layer, head, :]
                        true_mass_mean = np.mean(head_wise_activations[usable_labels == 1], axis=0)
                        false_mass_mean = np.mean(head_wise_activations[usable_labels == 0], axis=0)
                        com_directions.append(true_mass_mean - false_mass_mean)
                com_directions = np.array(com_directions)

                return com_directions

            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            head_wise_activations = deepcopy(all_head_wise_activations)
            head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)
            tuning_activations = deepcopy(all_head_wise_activations)
            tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h=num_heads)

            top_heads, probes = get_top_heads(head_wise_activations, labels, num_layers, num_heads, num_to_intervene)

            com_directions = get_com_directions(num_layers, num_heads, head_wise_activations,
                                                labels)

            interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads,com_directions)

            return interventions

        def preprocess_activate_dataset(self, dataset, system_prompt="You are a helpful, honest and concise assistant."):
            """
            Preprocesses the dataset and retrieves activations for the model.

            Args:
            dataset: Input dataset.
            system_prompt (str): System prompt for the model.

            Returns:
            all_head_wise_activations: All head-wise activations.
            """
            self.system_prompt = system_prompt

            def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
                if 'llama-3' in self.model_file.lower():
                    if model_output:
                        con = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": model_output}
                        ]
                        return torch.tensor(tokenizer.apply_chat_template(con)[:-5]).unsqueeze(0)
                    else:
                        con = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": instruction},
                        ]
                        return torch.tensor(tokenizer.apply_chat_template(con)).unsqueeze(0)
                else:
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
                    dialog_tokens = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                    return torch.tensor(tokenizer(dialog_tokens).input_ids).unsqueeze(0)

            def data_preprocess(dataset):
                all_prompts = []
                for i in range(len(dataset)):
                    question = dataset[i]['question']

                    pos_answer = dataset[i]['answer_matching_behavior']
                    pos_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, question, pos_answer)
                    all_prompts.append(pos_tokens)

                    neg_answer = dataset[i]['answer_not_matching_behavior']
                    neg_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, question, neg_answer)
                    all_prompts.append(neg_tokens)

                return all_prompts

            def get_llama_activations_bau(model, prompt):
                HEADS = [f"model.layers.{i}.block.self_attn.attn.head_out" for i in range(model.config.num_hidden_layers)]
                MLPS = [f"model.layers.{i}.block.mlp" for i in range(model.config.num_hidden_layers)]

                with torch.no_grad():
                    prompt = prompt.to(model.device)
                    with TraceDict(model, HEADS + MLPS) as ret:
                        output = model(prompt, output_hidden_states=True)
                    hidden_states = output.hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze().to(torch.float16).detach().cpu().numpy()
                    head_wise_hidden_states = [ret[head].output.squeeze().to(torch.float16).detach().cpu() for head in HEADS]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                    mlp_wise_hidden_states = [ret[mlp].output.squeeze().to(torch.float16).detach().cpu() for mlp in MLPS]
                    mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

                return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

            prompts = data_preprocess(dataset)

            all_layer_wise_activations = []
            all_head_wise_activations = []

            for prompt in tqdm(prompts):
                layer_wise_activation, head_wise_activation, _ = get_llama_activations_bau(self.model, prompt)
                all_layer_wise_activations.append(layer_wise_activation[:, -1, :])
                all_head_wise_activations.append(head_wise_activation[:, -1, :])

            return all_head_wise_activations

        def set_activate(self, interventions, alpha):
            """
            Sets the activations for the model based on interventions.

            Args:
            interventions: Dictionary of interventions.
            alpha: Activation strength.
            """
            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            for head_out_name, list_int_vec in interventions.items():
                layer_no = int(head_out_name.split('.')[2])
                displacement = np.zeros((num_heads, int(self.model.model.config.hidden_size / num_heads)))
                for head_no, head_vec, std in list_int_vec:
                    displacement[head_no] = alpha * std * head_vec
                device = self.model.model.layers[layer_no].block.self_attn.attn.o_proj.weight.device.index
                displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                if '70B' in self.model_file:
                    bias_tobe = F.linear(displacement.to(torch.bfloat16), self.weight_cache[layer_no]).to(device)
                else:
                    bias_tobe = F.linear(displacement.to(torch.float16), self.model.model.layers[layer_no].block.self_attn.attn.o_proj.weight).to(device)
                self.model.model.layers[layer_no].block.self_attn.attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

    model = LlamaLM()
    model.reset_all()
    return model, model.tokenizer
