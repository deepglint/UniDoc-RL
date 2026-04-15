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
import numpy as np
import os


def dcg(relevance_scores):
    """
    Compute discounted cumulative gain (DCG).

    Args:
        relevance_scores: Relevance score for each retrieved document.

    Returns:
        DCG value.
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value



def ndcg(rerank_image_basename_lists, golden_answer_list):
    if len(rerank_image_basename_lists) == 0:
        return 0.0

    top1_image_list = []
    top2_image_list = []
    top3_image_list = []
    for rerank_image_basename_list in rerank_image_basename_lists:
        if len(rerank_image_basename_list) > 0:
            top1_image_list.append(sorted(rerank_image_basename_list[0]))
        if len(rerank_image_basename_list) > 1:
            top2_image_list.append(sorted(rerank_image_basename_list[1]))
        if len(rerank_image_basename_list) > 2:
            top3_image_list.append(sorted(rerank_image_basename_list[2]))
    
    sorted_docs = top1_image_list + top2_image_list + top3_image_list
    
    new_sorted_docs = []
    for doc in sorted_docs:
        if doc not in new_sorted_docs:
            new_sorted_docs.append(doc)
    sorted_docs = new_sorted_docs
    
    # Map each document to a binary relevance score.
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    
    # Compute DCG.
    dcg_value = dcg(relevance_scores)
    
    # Compute IDCG, where all relevant documents are ranked first.
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
        
    idcg_value = dcg(ideal_relevance_scores)
    
    # Avoid division by zero.
    if idcg_value == 0:
        return 0.0
    
    # Compute NDCG.
    ndcg_value = dcg_value / idcg_value
    if ndcg_value >1.0:
        print(sorted_docs)
        print(golden_answer_list)
    return ndcg_value
    


def get_rerank_score(select_image_list, rerank_image_lists, golden_image_list):
    assert len(select_image_list) == len(rerank_image_lists)
    if len(select_image_list) == 0:
        return 0.0
    score = []
    for select_image, rerank_image_list in zip(select_image_list, rerank_image_lists):
        if len(rerank_image_list) == 0:
            score.append(0.0)
            continue
        golden_image = rerank_image_list[0]
        for image in rerank_image_list:
            if image in golden_image_list:
                golden_image = image
                break 
        
        if select_image == golden_image:
            score.append(1.0)
        else:
            score.append(0.0)
    average_score = sum(score) / len(score)
    return average_score


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




def compute_iou(box1, box2):
    try:
        x1, y1, x2, y2 = box1
        x1_g, y1_g, x2_g, y2_g = box2

        xi1 = max(x1, x1_g)
        yi1 = max(y1, y1_g)
        xi2 = min(x2, x2_g)
        yi2 = min(y2, y2_g)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_g - x1_g) * (y2_g - y1_g)

        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area
    except:
        return 0.0

def compute_iou_f1(R, num_predictions, iou_threshold=0.5, eps=1e-8):
    """
    Compute F1 from IoU rewards and the number of predicted boxes.
    
    Args:
        R (list of float): IoU reward for each ground-truth box.
        num_predictions (int): Total number of predicted boxes.
        iou_threshold (float): IoU threshold used to count true positives.
        eps (float): Small value added to avoid division by zero.
    
    Returns:
        f1 (float): F1 score.
        recall (float)
        precision (float)
    """

    R = np.array(R)
    tp = np.sum(R >= iou_threshold)  # Number of true positives.
    n = len(R)                       # Number of ground-truth boxes.

    # Avoid division by zero by adding `eps` to the denominator.
    recall = tp / (n + eps)
    precision = tp / (num_predictions + eps)

    # Compute F1 with the same zero-division safeguard.
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return f1
    

def compute_roi_scores(retrievaled_images, evidence_bbox_lists, golden_bbox_lists, golden_crop_lists):
    assert len(retrievaled_images)==len(evidence_bbox_lists)
    if len(retrievaled_images) == 0:
        return 0.0
    
    scores = []
    for retrievaled_image, evidence_bbox_list in zip(retrievaled_images, evidence_bbox_lists):
        if retrievaled_image not in golden_bbox_lists:
            golden_bbox_list = []
        elif golden_crop_lists[retrievaled_image]==False:
            golden_bbox_list = []
        else:
            golden_bbox_list = golden_bbox_lists[retrievaled_image]
    
        if len(golden_bbox_list)==0 and len(evidence_bbox_list)==0:
            mean_iou = 1.0
        elif len(golden_bbox_list)==0:
            mean_iou = 0.0
        elif len(evidence_bbox_list)==0:
            mean_iou = 0.0
        else:
            max_ious = []
            for gt_box in golden_bbox_list:
                best_iou = 0.0
                for pred_box in evidence_bbox_list:
                    iou = compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                max_ious.append(best_iou)
            mean_iou = compute_iou_f1(max_ious, len(evidence_bbox_list))
        
        scores.append(mean_iou)
    
    return sum(scores) / len(scores)


class RMManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,rm_url="http://0.0.0.0:8003/eval") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
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

        reward_tensor = torch.zeros((len(data), 5, data.batch['responses'].shape[-1]), dtype=torch.float32)
       
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

            format_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if format_score > 0.5:
                data_to_be_eval.append(data_eval[i])
        
        if len(data_to_be_eval) > 0:
            request_data_to_be_eval = dict(
                bs=300,
                prompts=data_to_be_eval
            )
            prompts_json = json.dumps(request_data_to_be_eval)
            print("=====================eval model start=====================")
            response = requests.post(self.rm_url, json=prompts_json)
            eval_results = response.json()
            print("=====================eval model end=====================")
        
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

            format_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            roi_score = 0.0
            ndcg_score = 0.0
            rerank_score = 0.0
            model_eval_score = 0.0
            final_score = format_score
            
            if format_score > 0:
                reference_pages = extra_info['reference_page'].tolist() if extra_info and 'reference_page' in extra_info else []
                reference_images_basename_list = [evidence_page for evidence_page in reference_pages]
                
                sorted_reference_images_basename_list = sorted(reference_images_basename_list)
                
                rerank_image_basename_lists = []
                rerank_image_lists = data_item.non_tensor_batch['rerank_image_lists']
                for rerank_image_list in rerank_image_lists:
                    rerank_image_basename_list = [os.path.basename(item.rstrip('/')).split(".jpg")[0] for item in rerank_image_list]
                    rerank_image_basename_lists.append(rerank_image_basename_list)
                
                
                ndcg_score = ndcg(rerank_image_basename_lists, sorted_reference_images_basename_list)
              
                
                selected_images_basename_list = [os.path.basename(item.rstrip('/')).split(".jpg")[0] for item in data_item.non_tensor_batch['selected_images']]
                
                rerank_score = get_rerank_score(selected_images_basename_list, rerank_image_basename_lists, reference_images_basename_list)
                
                evidence_bbox_lists = [evidence_bbox for evidence_bbox in data_item.non_tensor_batch['evidence_bbox_lists']]
                golden_bbox_lists = json.loads(data_item.non_tensor_batch['bboxes'])
                golden_crop_lists = json.loads(data_item.non_tensor_batch['crop_required'])
                
                
                roi_score = compute_roi_scores(selected_images_basename_list, evidence_bbox_lists, golden_bbox_lists, golden_crop_lists)
                
                if format_score > 0.5:
                    model_eval_score = eval_results.pop(0)
                    final_score = 0.6*model_eval_score + 0.1*ndcg_score + 0.1*rerank_score + 0.1*roi_score + 0.1*format_score
                else:
                    final_score = 0.1*ndcg_score + 0.1*rerank_score  + 0.1*roi_score + 0.1*format_score
            
            else:
                print("==================================================")
                print("[response]", response_str.replace("<|image_pad|>",""))
                
                
            reward_tensor[i, 0, valid_response_length - 1] = final_score
            reward_tensor[i, 1, valid_response_length - 1] = ndcg_score
            reward_tensor[i, 2, valid_response_length - 1] = rerank_score
            reward_tensor[i, 3, valid_response_length - 1] = roi_score
            reward_tensor[i, 4, valid_response_length - 1] = model_eval_score
            

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", final_score)

        return reward_tensor