import torch
import re
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from transformers.image_processing_base import BatchFeature
from PIL import Image
from tqdm import tqdm
import json
import traceback

invalid_prompt = """Your previous action is invalid. Please strictly follow the steps below to answer the user's question:

1. **Reason First**  
   Upon receiving any new information, you **must** first perform reasoning within <think> and </think> tags to determine whether you already have sufficient information to answer the user's question directly.

2. **Initiate a Search (if needed)**  
   If reasoning indicates that necessary information is missing, initiate an image search using <search>query</search>. The system will return a set of images (indexed starting from 0).

3. **Image Analysis and Selection**  
   - Within the <think> tags, **analyze each image one by one**, focusing on whether it contains **text, charts, tables, labels, or other key visual information** relevant to the query.  
   - Evaluate the completeness and relevance of information in each image, and **select the single image most helpful for answering the query**.  
   - Explicitly specify the index of the chosen image in a <select> tag (e.g., <select>2</select>).

4. **Detailed Image Content Analysis**  
   **After selecting an image, proceed with this two-stage analysis:**
   
   ### 4.1 Initial ROI Identification and Quality Assessment
   Within `<think>` tags:
   - Identify specific regions of interest (ROI) with coordinates `[x1, y1, x2, y2]` relevant to the query
   - For each region, summarize its key semantic content
   - **Assess readability**: Determine if content is clear or needs magnification
   
   ### 4.2 Content Extraction Decision
   Based on quality assessment:
   - **If content is readable**: Extract semantic meaning and output `<information>Answer based on this image</information>`
   - **If content needs magnification**: Request specific regions using `<bbox>[[x1, y1, x2, y2], ...]</bbox>` and pause
     *Magnification is needed only when enlarging the ROI of the selected image might help answer the query more clearly and accurately.*

5. **Magnified Image Processing (If Applicable)**  
   **Only execute if magnification was requested in Step 4:**
   - Upon receiving magnified image(s), analyze within `<think>` tags
   - Extract relevant information from the high-resolution regions
   - Output findings in `<information>Based on magnified image analysis</information>`

6. **Final Answer Determination**  
   Within `<think>` tags, evaluate:
   - Is current information sufficient to answer the original question?
   - What critical information (if any) remains missing?
   
   **Output exactly one:**
   - If insufficient: `<search>precise follow-up query</search>`
   - If sufficient: `<answer>direct final answer</answer>`

7. **Iterative Refinement**  
   Repeat Steps 3-6 as needed until sufficient information is gathered to provide a complete and accurate final answer.

**CRITICAL CLARIFICATION ON MAGNIFICATION:**
- Request magnification ONLY when enlarging the specific region of the selected image might help answer the query more clearly and accurately
- Use `<bbox>` coordinates to specify EXACT regions that need magnification
- After magnification, base your analysis SOLELY on the magnified region

Always adhere strictly to this sequential workflow to ensure methodical reasoning, justified image selection, and precise information extraction.
"""

def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)


    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int 
    num_gpus: int
    search_url: Optional[str] = None

class LLMGenerationManager:
    def __init__(
        self,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _postprocess_responses_first(self,batch):
        
        responses_str = self.tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        responses_str = ["<search>"+item.split('Question: ')[1].split(' \n\nassistant\n')[0]+"</search>" for item in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
        

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search operation or answer operation."""
        
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        
        def extract_tags(text):
            # Match supported tagged blocks in the model output.
            pattern = r"<(answer|search|think|information|bbox|select)>(.*?)</\1>"
            # Collect all matched blocks.
            matches = re.findall(pattern, text, re.DOTALL)
            # Rebuild the filtered response string.
            result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
            return result

        responses_str = [extract_tags(resp) + self.tokenizer.eos_token for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List, rollings, search_queries: List) -> Tuple[torch.Tensor, List[str], List[Dict[str, List[Any]]], List[BatchFeature]]:
        """Process next observations from environment."""
        next_obs_str = []
        multi_modal_data = []
        multi_modal_inputs = []
        merge_length = self.processor.image_processor.merge_size**2
        
        # print(f"------------------next_obs number: {len(next_obs)}--------------------------------")
        
        for idx, obs_item in enumerate(next_obs):
            try:
                # invalid
                if isinstance(obs_item,str):
                    next_obs_str.append(obs_item)
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
                elif isinstance(obs_item, list) and len(obs_item) == 0:
                    next_obs_str.append(f'\n<|im_start|>user\n{invalid_prompt}<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
                # invalid
                elif isinstance(obs_item, list) and not isinstance(obs_item[0],dict) and len(self.select_images[idx]) == 0:
                    print("------------------1--------------------")
                    next_obs_str.append(f'\n<|im_start|>user\n{invalid_prompt}<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
                # Crop the selected image region.
                elif isinstance(obs_item,list) and not isinstance(obs_item[0],dict):
                    try:
                        if isinstance(obs_item[0], int):
                            obs_item = [obs_item]
                        input_bbox_list = obs_item
                        latest_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][-1]
                        width, height = latest_image.size
                        raw_images_crop = Image.open(self.select_images[idx][-1])
                        raw_width, raw_height = raw_images_crop.size
                        
                        cropped_images = []
                        cropped_bbox = []
                        total_height = 0
                        max_width = 0
                        for bbox in input_bbox_list:
                            if self.is_validation:
                                bbox = [bbox[0]-28, bbox[1]-28, bbox[2]+28, bbox[3]+28]
                            crop_area = [int(raw_width * bbox[0] / width), int(raw_height * bbox[1] / height), int(raw_width * bbox[2] / width), int(raw_height * bbox[3] / height)]
                            crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
                            
                            relative_crop_area = [int(bbox[0]/width*1000), int(bbox[1]/height*1000), min(1000, int(bbox[2]/width*1000)), min(1000, int(bbox[3]/height*1000))]
                                                
                            cropped_bbox.append(relative_crop_area)
                            
                            if crop_area[0]>crop_area[2] or crop_area[1]>crop_area[3]:
                                print(f"crop_area: {crop_area}, raw_width: {raw_width}, raw_height: {raw_height}")
                            
                            if crop_area[0]>=crop_area[2]:
                                crop_area[0] = 0
                            if crop_area[1]>=crop_area[3]:
                                crop_area[1] = 0
                            
                            cropped_image = raw_images_crop.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))
                            cropped_images.append(cropped_image)
                            total_height += cropped_image.height
                            max_width = max(max_width, cropped_image.width)

                        new_image = Image.new('RGB', (max_width, total_height))
                        y_offset = 0
                        for img in cropped_images:
                            new_image.paste(img, (0, y_offset))
                            y_offset += img.height
                        
                        input_images_list = [new_image]
                        raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]
                        
                        image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')
                        multi_modal_inputs.append(image_inputs)
                        image_grid_thw = image_inputs['image_grid_thw']
                        obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                        raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
                        obs_str = '\n<|im_start|>user\nThe cropped image is:\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                        next_obs_str.append(obs_str)   
                        multi_modal_data.append({'image': raw_images_list})
                        
                        if len(self.evidence_bbox_lists[idx]) > 0:
                            if len(self.evidence_bbox_lists[idx][-1]) == 0:
                                self.evidence_bbox_lists[idx][-1] = cropped_bbox
                                
                                
                    except Exception as e:
                        print(f"Image cropping failed due to the following error: {str(e)}")
                        print(obs_item)
                        # print("Stack trace:")
                        #traceback.print_exc()
                        next_obs_str.append(f'\n<|im_start|>user\nImage cropping failed. bbox_list is a list containing multiple bounding boxes, each in the format [x_min, y_min, x_max, y_max], where all coordinates are integer pixel values satisfying 0 ≤ x_min < x_max and 0 ≤ y_min < y_max.<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict())) 
                
                # Process the retrieved candidate images.
                elif isinstance(obs_item,list) and isinstance(obs_item[0],dict):
                    
                    img_file_list = [item['image_file'] for item in obs_item]
                    new_img_file_list = []
                    for image_item in img_file_list:
                        if image_item not in self.select_images[idx]:
                            new_img_file_list.append(image_item)
                    # Keep the top-3 retrieved images.
                    input_images_list = new_img_file_list[:3]
                    self.rerank_image_lists[idx].append(input_images_list)
                        
                    raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                    multi_modal_data.append({'image': raw_images_list})
                    image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

                    multi_modal_inputs.append(image_inputs)
                    image_grid_thw = image_inputs['image_grid_thw']
                    
                    obs_str = "The list of retrieved images is as follows:\n"
                    for image_idx, image_grid_thw_item in enumerate(image_grid_thw):
                        if image_idx >0 :
                            obs_str += "\n"
                        obs_str += f"Image {image_idx}:"
                        obs_str += f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>"
                
                    obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                    next_obs_str.append(obs_str)
                    
                    self.select_images[idx].append(input_images_list[0])
                    self.retrievaled_images[idx].append(input_images_list[0])
                    self.evidence_bbox_lists[idx].append([])
                
                # Select the most relevant image.
                elif isinstance(obs_item,int):
                    select_image_index = obs_item
                    if len(self.rerank_image_lists[idx])==0:
                        next_obs_str.append(f'\n<|im_start|>user\n{invalid_prompt}<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
                        continue
                    
                    last_search_images = self.rerank_image_lists[idx][-1]
                    
                    if len(last_search_images)==0:
                        next_obs_str.append(f'\n<|im_start|>user\n{invalid_prompt}<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
                        continue
                    
                    if select_image_index == -1 or select_image_index>=len(last_search_images):
                        self.select_images[idx][-1] = last_search_images[0]
                        input_images_list = [last_search_images[0]]
                    else:
                        self.select_images[idx][-1] = last_search_images[select_image_index]
                        input_images_list = [last_search_images[select_image_index]]
                    
                    raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                    multi_modal_data.append({'image': raw_images_list})
                    image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

                    multi_modal_inputs.append(image_inputs)
                    image_grid_thw = image_inputs['image_grid_thw']
                    
                    obs_str = "The selected image is:\n"
                    obs_str += ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                    raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
                    obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                    next_obs_str.append(obs_str)
                    
                else:

                    raise ValueError('invalid observation')
                
            except Exception as e:
                print(f"Error in _process_next_obs: {str(e)}")
                print(traceback.format_exc())
                print(obs_item)
                next_obs_str.append(f'\n<|im_start|>user\n{invalid_prompt}<|im_end|>\n<|im_start|>assistant\n')
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
                continue
        
        next_obs_ids = self.tokenizer(
            next_obs_str, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
    
    def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data:list, next_obs_multi_modal_inputs:list):
        if not 'multi_modal_inputs' in rollings.non_tensor_batch.keys():

            rollings.non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
            for idx, item in enumerate(next_obs_multi_modal_inputs):
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = item

            rollings.non_tensor_batch['multi_modal_data'] = np.array(next_obs_multi_modal_data, dtype=object)

        else:
            for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
                if len(multi_modal_data_item['image']) > 0:
                    # data
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(multi_modal_data_item['image'])
                    if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], next_obs_multi_modal_inputs[idx]['pixel_values']),dim=0)
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], next_obs_multi_modal_inputs[idx]['image_grid_thw']),dim=0)
                    else:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_obs_multi_modal_inputs[idx]['pixel_values']
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_obs_multi_modal_inputs[idx]['image_grid_thw']
        return rollings
        

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        if next_obs_ids.shape[1] != 0:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        }, rollings.non_tensor_batch)

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: Optional[torch.Tensor] = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None and next_obs_ids.shape[1] != 0:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        padded_non_tensor_batch = {}

        padded_ids = self.tokenizer(
            ['<|im_start|>user\nHi, who are u?<|im_end|>\n<|im_start|>assistant\n'], 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        padded_ids = padded_ids[0]

        pad_input_ids = torch.full_like(active_batch.batch['input_ids'][0], 151643, dtype=torch.int64)
        pad_input_ids[:len(padded_ids)] = padded_ids
        pad_attention_mask = self.tensor_fn.create_attention_mask(pad_input_ids)
        pad_input_ids = pad_input_ids.unsqueeze(0)
        pad_attention_mask = pad_attention_mask.unsqueeze(0)
        pad_position_ids = self.tensor_fn.create_position_ids(pad_attention_mask)
        
        padded_batch['attention_mask'] = torch.cat([active_batch.batch['attention_mask'], pad_attention_mask.repeat(padding_size, *[1] * (len(active_batch.batch['attention_mask'].shape) - 1))], dim=0)
        padded_batch['input_ids'] = torch.cat([active_batch.batch['input_ids'], pad_input_ids.repeat(padding_size, *[1] * (len(active_batch.batch['input_ids'].shape) - 1))], dim=0)
        padded_batch['position_ids'] = torch.cat([active_batch.batch['position_ids'], pad_position_ids.repeat(padding_size, *[1] * (len(active_batch.batch['position_ids'].shape) - 1))], dim=0)
        

        for k, v in active_batch.non_tensor_batch.items():
            pad_non_tensor_item = np.empty(padding_size, dtype=object)
            if k == 'raw_prompt_ids':
                list_ids = padded_ids.tolist()
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = list_ids
            elif k == 'multi_modal_inputs':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {}
            elif k == 'multi_modal_data':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {'image': []}
            else:
                default_value = v[0] if len(v) > 0 else None
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = default_value
            padded_non_tensor_batch[k] = np.concatenate([v, pad_non_tensor_item])
                
        padded_active_batch = DataProto.from_dict(padded_batch, padded_non_tensor_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _raw_prompt_ids(self, rollings):
        new_raw_prompt_ids = []
        rollings.batch['input_ids'] = rollings.batch['input_ids'].long()
        raw_next_obs_ids = [ids[mask == 1].tolist() for ids, mask in zip(np.array(rollings.batch['input_ids']),  np.array(rollings.batch['attention_mask']))]
        def replace_consecutive_elements(arr, target):
            result = []
            i = 0
            while i < len(arr):
                if arr[i] == target:
                    result.append(target)
                    while i + 1 < len(arr) and arr[i + 1] == target:
                        i += 1
                else:
                    result.append(arr[i])
                i += 1
            return result
        raw_next_obs_ids = [replace_consecutive_elements(row,151655) for row in raw_next_obs_ids]
        raw_next_obs_ids = np.array(raw_next_obs_ids, dtype=object)
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_next_obs_ids
        return rollings

    def deactivate_batch(self, active_mask,rollings):
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        max_model_len = 10240
        curr_active_mask = torch.tensor([len(raw_prompt_ids_item) < max_model_len for raw_prompt_ids_item in raw_prompt_ids], dtype=torch.bool)
        active_mask = active_mask * curr_active_mask
        return active_mask

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']

        self.retrievaled_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        
        self.rerank_image_lists = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        
        self.select_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        
        self.evidence_bbox_lists = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        meta_info = {}


        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            
            
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)
            if not active_mask.sum():
                break
            
            if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                rollings_active = DataProto.from_dict(
                    tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                    non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                )
            else:
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })

            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
    
            
            # Execute in environment and process observations
            next_obs, dones, search_queries = self.execute_predictions(responses_str, self.tokenizer.pad_token, rollings, active_mask)
            
            
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            next_obs_ids, next_obs_str, next_obs_multi_modal_data, next_obs_multi_modal_inputs = self._process_next_obs(next_obs, rollings, search_queries)
            
            if next_obs_ids is None:
               continue
            
            rollings = self._concat_multi_modal_data(
                rollings,
                next_obs_multi_modal_data,
                next_obs_multi_modal_inputs
            )
            
            # Update states            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )



        # final LLM rollout
        if active_mask.sum():

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)

            if active_mask.sum():

                if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })

                gen_output = self._generate_with_gpu_padding(rollings_active)

                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                # # Execute in environment and process observations
                _, dones, _ = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, rollings, active_mask, do_search=False
                )

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())

                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # =================== raw prompt ids ===================
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
        # rollings.non_tensor_batch.pop('raw_prompt_ids')
        
        if not self.is_validation:
            rollings, original_right_side = self._add_noisy_multi_modal_data(rollings, original_right_side)
        ### check again
        
        retrievaled_images_array = np.empty(len(self.retrievaled_images), dtype=object)
        for idx in range(len(self.retrievaled_images)):
            retrievaled_images_array[idx] = self.retrievaled_images[idx]
        
        selected_images_array = np.empty(len(self.select_images), dtype=object)
        for idx in range(len(self.select_images)):
            selected_images_array[idx] = self.select_images[idx]
        
        rerank_image_lists_array = np.empty(len(self.rerank_image_lists), dtype=object)
        for idx in range(len(self.rerank_image_lists)):
            rerank_image_lists_array[idx] = self.rerank_image_lists[idx]
        
        
        evidence_bbox_array = np.empty(len(self.evidence_bbox_lists), dtype=object)
        for idx in range(len(self.evidence_bbox_lists)):
            evidence_bbox_array[idx] = self.evidence_bbox_lists[idx]
            
        rollings.non_tensor_batch['retrievaled_images'] = retrievaled_images_array
        rollings.non_tensor_batch['rerank_image_lists'] = rerank_image_lists_array
        rollings.non_tensor_batch['selected_images'] = selected_images_array
        rollings.non_tensor_batch['evidence_bbox_lists'] = evidence_bbox_array
        
        
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollings)
    
    def _add_noisy_multi_modal_data(self, rollings, original_right_side):
        print("------------------add noisy multi modal data-----------------------")
        
        image_padded = Image.new('RGB', (64, 64), (0, 0, 0))
        image_padded = process_image(image_padded, 256*256, 128*128)
        image_inputs = self.processor.image_processor([image_padded], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size**2
#       
        padded_str = f"\n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|>"+"<|im_end|>"
        
        padded_str_list = []
        for idx, multi_modal_item in enumerate(rollings.non_tensor_batch['multi_modal_data']):
            if len(multi_modal_item['image']) == 0:
                padded_str_list.append(padded_str)
                rollings.non_tensor_batch['multi_modal_data'][idx]['image'].append(image_padded)
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = image_inputs
            else:
                padded_str_list.append('')
            
        padded_ids = self.tokenizer(
            padded_str_list, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        original_right_side = self._update_right_side(
            original_right_side,
            padded_ids
        )
        return rollings, original_right_side


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollings) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output,rollings.non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, rollings, active_mask=None, do_search=True) -> Tuple[List[Any], List[int], List[str]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        if active_mask is None:
            active_mask = [True] * len(cur_actions)
        
        next_obs, dones = [], []
        
        bbox_list = [content for action, content in zip(cur_actions, contents) if action == 'bbox']
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        select_image_index_list = [content for action, content in zip(cur_actions, contents) if action == 'select']

        
        if do_search:
            if len(search_queries) > 0:
                if not self.config.search_url:
                    raise ValueError("`search_url` must be configured when search actions are enabled.")
                batch_size = 100
                search_results = []
                for i in range(0, len(search_queries), batch_size):
                    batch_queries = search_queries[i:i + batch_size]
                    response = requests.get(self.config.search_url, params={"queries": batch_queries})
                    search_results_single_batch = response.json()
                    search_results.extend(search_results_single_batch)
                assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
            else:
                search_results = []
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
        
        
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            try:
                if len(rollings.non_tensor_batch['multi_modal_data'][i]['image']) > 22:
                    print("=================================================")
                    print(len(rollings.non_tensor_batch['multi_modal_data'][i]['image']))
                    print("=================================================")
                    next_obs.append('')
                    dones.append(1)
                    continue
            except:
                pass
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                elif action == 'search':
                    next_obs.append(search_results.pop(0))
                    dones.append(0)
                    
                elif action == 'bbox':
                    try:
                        bbox_value = json.loads(bbox_list.pop(0))
                        next_obs.append(bbox_value)
                    except:
                        next_obs.append(f'Your previous action is invalid. The bbox_list is a list containing multiple bounding boxes, each in the format [x_min, y_min, x_max, y_max], where all coordinates are integer pixel values satisfying 0 ≤ x_min < x_max and 0 ≤ y_min < y_max.')
                    dones.append(0)
                elif action == 'select':
                    try:
                        select_image_index = int(select_image_index_list.pop(0))
                        next_obs.append(select_image_index)
                    except:
                        next_obs.append(-1)
                    dones.append(0)
                else:
                    print("----------------3--------------------------")
                    next_obs.append(f'\n<|im_start|>user\n{invalid_prompt}<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
            
        assert len(search_results) == 0

        return next_obs, dones, search_queries

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[Optional[str]], List[str]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        action_pattern = re.compile(r'<(search|answer|select|bbox)>(.*?)</\1>', re.DOTALL)
        
        actions = []
        contents = []

        for i, prediction in enumerate(predictions):
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction type at index {i}: {type(prediction)}")

            # Match the first supported action block.
            action_match = action_pattern.search(prediction)
            if action_match:
                action = action_match.group(1)
                content = action_match.group(2).strip()
            else:
                action = None
                content = ''
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents