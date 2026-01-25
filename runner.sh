
source .venv/bin/activate




datasets=(
'Algebra-and-Trigonometry-2e-WEB.json'
'Biology2e-WEB.json'
'Chemistry2e-WEB.json'
'College_Physics_2e-WEB_7Zesafu.json'
'Introduction_To_Computer_Science-WEB.json'
'Psychology2e_WEB.json'
)


model="gpt-4.1-,omo"
remote="openai"


echo "Generating Open-Ended datasets"
fldr=./data-mmlu/generative_mmlu_pro/


out_fldr=./data-mmlu/oe-gpt4.1
for dataset in "${datasets[@]}"; do
  python generate_novel_open.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model} --sample_count=800
done


# evaluating basline MMLU-pro benchmark accuracy
python inspect_eval_open_mmlu.py --base_dir=./data-mmlu/mmlu_pro
# evaluating generative MMLU-pro benchmark accuracy
python inspect_eval_open_mmlu.py --base_dir=${out_fldr}


