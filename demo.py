import os
import glob
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import shutil


input_folder = './demo/source_data'  # folder to place 1 or more pdf files
chunked_folder = './demo/chunked'
open_dataset_folder = './demo/oe_questions'
log_folder = './demo/logs'

if os.path.exists(chunked_folder):
    shutil.rmtree(chunked_folder)
if os.path.exists(open_dataset_folder):
    shutil.rmtree(open_dataset_folder)
if os.path.exists(log_folder):
    shutil.rmtree(log_folder)

import convert_pdf
convert_pdf.convert_pdfs_to_qa_chunks(input_folder=input_folder, output_folder=chunked_folder, sample_count=4, chunk_size=2000)


import generate_novel_open
fns = [fn for fn in os.listdir(chunked_folder) if fn.endswith('.json')]
for fn in fns:

    generate_novel_open.generate(
            input_filepath=os.path.join(chunked_folder, fn),
            output_filepath=os.path.join(open_dataset_folder, fn),
            remote='pn131285:8443',
            model='gpt120b',
            async_flag=True,
            sample_count=32
        )

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()



import inspect_open
models_config = [
    {"model_name": "v_llm/openai/gpt-oss-20b", "base_url": "https://pn125916.nist.gov:8443/v1", "api_key": os.getenv("VLLM_API_KEY")},
    {"model_name": "v_llm/meta-llama/Llama-3.1-8B-Instruct", "base_url": "https://pn125916.nist.gov:8444/v1", "api_key": os.getenv("VLLM_API_KEY")},
    {"model_name": "v_llm/gpt120b", "base_url": "https://pn131285.nist.gov:8443/v1", "api_key": os.getenv("VLLM_API_KEY")},
    # {"model_name": "v_llm/gpt-oss-120b", "base_url": "https://rchat.nist.gov/api", "api_key": os.getenv("RCHAT_API_KEY")},
    {"model_name": "v_llm/Llama-4-Maverick-17B-128E-Instruct-FP8", "base_url": "https://rchat.nist.gov/api", "api_key": os.getenv("RCHAT_API_KEY")}
  ]

inspect_open.run(base_dir=open_dataset_folder, models_config=models_config, log_dir=log_folder)

# Plot accuracy results from logs
model_accuracies = {}
for fp in glob.glob(os.path.join(log_folder, '*.json')):
    with open(fp, 'r') as f:
        data = json.load(f)
    model = data['eval']['model']
    accuracy = data['results']['scores'][0]['metrics']['accuracy']['value']
    model = model.replace('-FP8', '')
    model = model.replace('-17B-128E', '')
    model = model.replace('-Instruct', '')
    model = model.replace('openai/', '')
    model = model.replace('meta-llama/', '')
    model_accuracies[model] = accuracy

models = sorted(model_accuracies.keys())
accuracies = [model_accuracies[m] for m in models]

fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
bars = ax.bar(range(len(models)), accuracies)
ax.set_xticks(range(len(models)))
ax.set_xticklabels([m.replace('v_llm/', '') for m in models], rotation=15, ha='right')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.1)
ax.set_title('Model Accuracy on Open-Ended QA')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
accuracy_png = os.path.join(log_folder, 'accuracy.png')
plt.savefig(accuracy_png, dpi=150)
plt.close()

import sys
opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
subprocess.Popen([opener, accuracy_png])