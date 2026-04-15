# UniDoc-RL

<div align="center">

## Reinforcement Learning for Multi-turn Visual Retrieval and Region-grounded Reasoning on Long Documents

<p><strong>An academic-style open-source codebase for document-centric visual RAG</strong></p>

<p>
  <img src="https://img.shields.io/badge/status-research%20code-blue" alt="status">
  <img src="https://img.shields.io/badge/modality-document%20images-orange" alt="modality">
  <img src="https://img.shields.io/badge/training-RL%20%2B%20synthetic%20CoT-green" alt="training">
  <img src="https://img.shields.io/badge/backbone-Qwen%20VL%20%2F%20vLLM-purple" alt="backbone">
</p>

<p align="center">
  <img src="img/intro.png" width="88%" alt="UniDoc-RL teaser" />
</p>

</div>

## 🚀 Overview

- We introduce **UniDoc-RL**, a reinforcement learning framework for **multi-turn visual retrieval-augmented reasoning** over long and visually rich documents.
- We formulate document understanding as an iterative interaction process in which the model learns to **reason, retrieve, select, crop, and answer** rather than relying on single-shot page understanding.
- We release the **training framework** of UniDoc-RL, including multimodal retrieval, synthetic trajectory construction, reward evaluation, and RL optimization for tool-using vision-language agents.

In UniDoc-RL, the model interacts with an external environment through structured actions such as `<search>`, `<select>`, `<bbox>`, and `<answer>`. This design enables the agent to progressively gather evidence from coarse page-level retrieval to fine-grained region inspection, which is especially useful for charts, tables, dense text regions, and multi-page evidence aggregation.

<div align="center">
<p align="center">
  <img src="img/unidoc_main.png" width="92%" alt="UniDoc-RL framework" />
</p>
</div>

We emphasize two key properties of UniDoc-RL: **progressive evidence acquisition** and **region-grounded interaction**. The agent first searches over page images, then selects the most informative visual evidence, and finally performs targeted crop-and-zoom operations when the answer depends on local fine-grained content.

<div align="center">
<p align="center">
  <img src="img/case.png" width="82%" alt="UniDoc-RL case study" />
</p>
</div>


## ⚙️ Train Model with UniDoc-RL

### Training Dependencies

```bash
conda create -n unidoc_rl python=3.10
conda activate unidoc_rl

git clone https://github.com/deepglint/UniDoc-RL.git
cd UniDoc-RL
```


### Step1. Prepare Data.

#### Benchmark & Training Data

UniDoc-RL assumes a document QA setting where each sample contains a question, a reference answer, and document-level supervision signals used during retrieval and reward computation. In practice, the RL stage expects **Parquet** files, while the synthetic data pipeline typically starts from **JSON/JSONL** annotations.

At a minimum, each sample should provide:

- a unique sample id,
- the user question,
- the reference answer,
- document/page-level metadata,
- optional region-level supervision for crop-based rewards.

An example annotation is shown below:

```json
{
  "uid": "sample_000001",
  "query": "What is the reported top-1 accuracy in the ablation study?",
  "reference_answer": "84.7%",
  "meta_info": {
    "file_name": "example_document.pdf",
    "reference_page": [12],
    "source_type": "Text/Table",
    "query_type": "Single-Hop"
  }
}
```

For RL training, the final parquet file is expected to contain chat-style prompts together with image fields and auxiliary metadata used by the reward manager. In particular, the local dataset loader in `RLTrain/verl/utils/dataset/rl_dataset.py` reads:

- `prompt`: chat-format prompt,
- `images`: multimodal image inputs,
- `extra_info`: question and page metadata,
- reward-related fields such as ground-truth answers and, when available, region annotations.


### Step2. Build Training Corpus & Run Multi-Modal Search Engine.

UniDoc-RL trains against a retrieval environment built on document page images. You should first convert each document into page-level images and organize them into a corpus directory.

Install the dependencies for the retrieval toolkit:

```bash
cd tools
pip install -r requirements.txt
```

Build the retrieval index:

```bash
cd tools/search_engine
python ingestion.py --dataset_dir /path/to/your/corpus
```

Then launch the search engine API:

```bash
cd tools/search_engine
SEARCH_DATASET_DIR=/path/to/your/corpus \
SEARCH_PORT=9001 \
python search_engine_api.py
```

By default, the retrieval module uses a ColQwen-style vision-language embedding backend and returns top-$k$ relevant page images for each search action.


### Step3. Construct High-quality CoT & Learn Patterns via SFT.

Before RL, UniDoc-RL supports constructing high-quality multi-turn trajectories with the synthetic data pipeline in `synthetic_data/`. This stage can generate search, rerank, crop, and answer traces that are useful for bootstrapping the model with supervised fine-tuning.

Install the dependencies for synthetic trajectory generation:

```bash
cd synthetic_data
pip install -r requirements.txt
```

Example command:

```bash
cd synthetic_data
python generate_cot_data.py \
  --data-path /path/to/data.json \
  --output-path /path/to/output.jsonl \
  --search-engine-url http://127.0.0.1:9001/search \
  --layout-parser-url http://127.0.0.1:30000 \
  --llm-engine-url http://127.0.0.1:9009/v1 \
  --vllm-engine-url http://127.0.0.1:9000/v1 \
  --enable-rerank \
  --enable-crop
```

To convert the generated JSONL trajectories into the open-source ShareGPT-style multimodal format expected by LLaMA-Factory:

```bash
cd synthetic_data
python convert_to_llamafactory.py \
  --input-path /path/to/output.jsonl \
  --output-path ../LLaMA-Factory/data/unidoc_sft.json
```

The generated trajectories can then be converted into chat-style training data for SFT. In our framework, this stage teaches the model the basic interaction pattern of:

- reasoning before acting,
- issuing retrieval queries when evidence is missing,
- selecting the most relevant page image,
- requesting region magnification only when needed,
- answering after sufficient evidence has been collected.

#### SFT with LLaMA-Factory

This repository already includes a local copy of [LLaMA-Factory](./LLaMA-Factory), which can be used to run multimodal SFT on the trajectories generated above.

First, install the LLaMA-Factory dependencies:

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

Then convert your synthetic trajectories into a ShareGPT-style multimodal JSON file and place it at `LLaMA-Factory/data/unidoc_sft.json`. A minimal example is shown below:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Question: What is the reported top-1 accuracy in the ablation study?"
      },
      {
        "role": "assistant",
        "content": "<think>...</think>\n<search>...</search>"
      }
    ],
    "images": [
      "/path/to/page_12.png"
    ]
  }
]
```

The corresponding dataset entry has already been registered in `LLaMA-Factory/data/dataset_info.json` under the name `unidoc`, so in most cases you only need to prepare `unidoc_sft.json` with the `messages` and `images` fields.

For SFT with Qwen2.5-VL:

```bash
cd LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml
```

### Step4. Run RL Training with Qwen2.5-VL-Instruct.


#### Start Training

Install the dependencies for RL training:

```bash
cd RLTrain
pip install -r requirements.txt
```

If you use the model-based evaluator during training, start the evaluation service first:

```bash
cd tools/model_eval
pip install -r ../requirements.txt
EVAL_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
EVAL_PORT=8003 \
python evaluator_api.py
```

Set the required environment variables:

```bash
export MODEL_PATH=/path/to/your/sft-checkpoint
export TRAIN_FILE=/path/to/your/train.parquet
export VAL_FILE=/path/to/your/val.parquet
export SEARCH_URL=http://127.0.0.1:9001/search
export RM_URL=http://127.0.0.1:8003/eval
```

Then start training:

```bash
cd RLTrain
bash train.sh
```

The default launcher configures GRPO-style RL optimization with multi-turn rollouts, multimodal prompts, external retrieval, and model-based reward evaluation.


## 🙏 Acknowledgements

This work is implemented based on and inspired by several open-source projects, including [verl](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), [LlamaIndex](https://github.com/run-llama/llama_index), and the [Qwen-VL](https://github.com/QwenLM) ecosystem. We also thank the communities behind multimodal retrieval backbones such as ColQwen for making document-centric visual RAG research more accessible.


## 📝 Citation

If you find this repository useful, please cite the corresponding paper. The bibliographic information can be updated once the public paper entry is finalized.

```bibtex
@misc{unidocrl2026,
      title={UniDoc-RL: Reinforcement Learning for Multi-turn Visual Retrieval and Region-grounded Reasoning on Long Documents},
      author={Author List To Be Updated},
      year={2026},
      note={Project page and paper link will be updated.}
}
```
