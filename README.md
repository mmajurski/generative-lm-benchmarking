# Generative Benchmarking of LMs using Grounding Documents

## Overview
This code generates multiple-choice (MCQs) or Open-Ended (OE) questions from grounding datasets.

### Paper

This repository hosts the code used in the paper [https://arxiv.org/abs/2505.08905](https://arxiv.org/abs/2505.08905).

## Setup

### Step1: Install uv [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Step 2: Create a virtual env and install dependencies

```shell
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

## Usage

Setup a .env file with your OPENAI_API_KEY
```shell
OPENAI_API_KEY=sk-proj-##############
```

The default model to generate questions and grade responses is openai/gpt-4.1-nano


### Reformat the Squad Dataset into Questions

```shell
python generate_reformat_mcq.py --out_dataset_dir=./data-qa/ --sample_count=4
python generate_reformat_open.py --out_dataset_dir=./data-qa/ --sample_count=4
```

This reformats the human written questions in the Squad questions into a format containing all required information for an LM to use a stand alone benchmark.


### Generate New Synthetic Questions from the Squad Dataset

```shell
python generate_novel_mcq.py --out_dataset_dir=./data-qa/ --sample_count=4
python generate_novel_open.py --out_dataset_dir=./data-qa/ --sample_count=4
```

This generates new synthetic questions based on the contexts in the Squad dataset, creating a standalone LM benchmark.



### Evalute Questionss

```shell
python inspect_eval.py --dataset_fldr=./data-qa/
python inspect_eval_open.py --dataset_fldr=./data-qa/
```

This will run the generated evaluation benchmarks against the LM under test.

The script will print to console the resulting Inspect framework results

```shell
╭─ squad_mcq (1,654 samples): hf/meta-llama/Llama-3.1-8B-Instruct ─────────────╮
│ max_connections: 4                                        dataset: squad_mcq │
│                                                               scorer: choice │
│ total time:                  0:12:25                                         │
│ hf/meta-llama/Llama-3.1-8B…  262,404 tokens [I:                              │
│                            214,094, O: 48,310]                               │
│                                                                              │
│ accuracy: 0.503  stderr: 0.0123                                              │
│                                                                              │
│ Log: logs/2024-11-17T19-27-32-05-00_squad-mcq_NQcUF95Z2i262tsYuSAfCv.eval    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

After capturing various test results, correlation between novel and hand-curated questions on a topic can be determined.