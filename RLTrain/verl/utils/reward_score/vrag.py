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

import re

def remove_text_between_tags(text):
    # Remove the user message wrapper inserted by the chat template.
    pattern = r'<\|im_start\|>user.*?<\|im_end\|>'
    # Replace matched content with an empty string.
    result = re.sub(pattern, '', text)
    return result




def compute_format_reward_only(predict_str: str, ground_truth: str, extra_info) -> float:
    predict_str = remove_text_between_tags(predict_str)
    answer_pattern = re.compile(r'<answer>.*</answer>', re.DOTALL)
    search_pattern = re.compile(r'<search>.*</search>', re.DOTALL)
    select_pattern = re.compile(r'<select>.*</select>', re.DOTALL)
    information_pattern = re.compile(r'<information>.*</information>', re.DOTALL)
    answer_match = re.search(answer_pattern, predict_str)
    search_match = re.search(search_pattern, predict_str)
    select_match = re.search(select_pattern, predict_str)
    information_match = re.search(information_pattern, predict_str)
    if answer_match and search_match and select_match and information_match:
        return 1.0
    if search_match and select_match and information_match:
        return 0.5
    return 0.0