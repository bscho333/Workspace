seed=42
model="llava"
model_path="/home/bscho333/data/llava-v1.5-7b"
coco_path="/home/bscho333/data/coco"
# inference img
img_path="${coco_path}/val2014/"

anno_path="${coco_path}/annotations/instances_val2014.json"
log_path="./logs/inference"
out_path="./inference_results"

use_ritual=False
use_vcd=False
use_m3id=False

ritual_alpha_pos=3.0
ritual_alpha_neg=1.0
ritual_beta=0.1

experiment_index=0

num_eval_samples=500
max_new_tokens=64

export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1 --master_port 2222 inference.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--data_path ${img_path} \
--anno_path ${anno_path} \
--log_path ${log_path} \
--out_path ${out_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--ritual_alpha_pos ${ritual_alpha_pos} \
--ritual_alpha_neg ${ritual_alpha_neg} \
--ritual_beta ${ritual_beta} \
--num_eval_samples ${num_eval_samples} \
--max_new_tokens ${max_new_tokens} \
--experiment_index ${experiment_index}