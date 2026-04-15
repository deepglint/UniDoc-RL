# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json
import requests
import math
import numpy as np
import os
from time import sleep


def get_answer_from_predict_str(text):
    end_tag = '</answer>'
    start_tag = '<answer>'
    
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None  # Return `None` if `</answer>` is missing.
    
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None  # Return `None` if `<answer>` is missing.
    
    start_pos += len(start_tag)  # Skip the opening `<answer>` tag.
    return text[start_pos:end_pos]


class RMEvalManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,rm_url="http://0.0.0.0:8003/eval") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # Number of decoded samples to print.
        self.compute_score = compute_score or _default_compute_score
        self.rm_url = rm_url
       
    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        recall_list = []
        ndcg_list = []

        already_print_data_sources = {}

        data_eval = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            generated_answer = get_answer_from_predict_str(self.tokenizer.decode(valid_response_ids))
            if generated_answer is None:
                generated_answer = 'Please Judge False'
            data_eval.append(dict(
                query = extra_info['question'],
                generated_answer = generated_answer,
                reference_answer = data_item.non_tensor_batch['reward_model']['ground_truth']
            ))

        data_to_be_eval = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            #if score >0.0:
            data_to_be_eval.append(data_eval[i])
        
        if len(data_to_be_eval) > 0:
            bs = 200
            while True:
                try:
                    request_data_to_be_eval = dict(
                        bs=bs,
                        prompts=data_to_be_eval
                    )
                    prompts_json = json.dumps(request_data_to_be_eval)
                    print("=====================rm eval model start=====================")
                    response = requests.post(self.rm_url, json=prompts_json)
                    eval_results = response.json()
                    print("=====================rm eval model end=====================")
                    break
                except:
                    print("=====================rm eval model error=====================")
                bs = max(int(bs/2),5)
                sleep(5)
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
  
            score = eval_results.pop(0)


            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str.replace("<|image_pad|>",""))
                print("[ground_truth]", ground_truth)
                print("[score]", score)
                #print("retrievaled_images_basename_list",data_item.non_tensor_batch['retrievaled_images'])
                

        return reward_tensor, ndcg_list, recall_list
