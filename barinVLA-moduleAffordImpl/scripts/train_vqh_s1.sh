DEBUG=false
if [ "$DEBUG" = true ]; then
  GPUS=1
  PER_DEVICE_BATCH_SIZE=8
  wandb_enable=false
  ACCELERATE_ARGS="--num_machines 1 --num_processes ${GPUS} --mixed_precision=bf16 --dynamo_backend=no"
  num_workers=0
  save_freq=5
  steps=10
fi

# distributed settings
GPUS=${GPUS:-1}  # 修改为默认使用1个GPU
GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # 每个节点1个GPU
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
wandb_enable=${wandb_enable:-true}
num_workers=${num_workers:-0}
save_freq=${save_freq:-5000}

# set environments
source scripts/env.sh

# distritubed training
find_free_port() {
    while true; do
        port=$(shuf -i 20000-65535 -n 1)
        if ! netstat -tna | grep -q ":${port}.*LISTEN"; then
            echo $port
            break
        fi
    done
}
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(find_free_port)
ACCELERATE_ARGS=${ACCELERATE_ARGS:-"--num_machines 1 --num_processes 1 --mixed_precision=no --dynamo_backend=no"}  # 修改为单GPU配置

# dataset mapping
declare -A data_map
data_map["libero_spatial"]=libero/spatial_20250605
data_map["libero_object"]=libero/object_20250605
data_map["libero_goal"]=libero/goal_20250605
data_map["libero_10"]=libero/libero_10_no_noops_20250603

declare -A pretrained_map
pretrained_map["libero_spatial"]=/liujinxin/code/sj/dualAffordance/hume/checkpoints/Hume-System2
pretrained_map["libero_object"]=/liujinxin/code/sj/dualAffordance/hume/checkpoints/Hume-System2
pretrained_map["libero_goal"]=/liujinxin/code/sj/dualAffordance/hume/checkpoints/Hume-System2
pretrained_map["libero_10"]=/liujinxin/code/sj/dualAffordance/hume/checkpoints/Hume-System2

data_name=libero_goal
dataset=${data_map[$data_name]}
echo "dataset: ${dataset}"
cfg=$(echo $dataset | sed 's/^\([a-zA-Z]\+\).*/\1/')

# Hyper Parameters
s1_chunk_size=8
s2_chunk_size=16
vqh_chunk_size=1
s1_his_state_size=1

theta2=1.0
theta1=1.0
noise_slides_alp=0.0
noise_slides_eps=0.0
cache_s2_actions=

lr=5e-5
critic_lr=1e-5
actor_lr=1e-5
temp_lr=2e-5

steps=$((GPUS * 200000))  # 训练步数根据GPU数量自动调整

# exp names
train_args="${theta2:+--theta2=${theta2}} "\
"${theta1:+--theta1=${theta1}} "\
"${noise_slides_eps:+--noise_slides_eps=${noise_slides_eps}} "\
"${noise_slides_alp:+--noise_slides_alp=${noise_slides_alp}} "\
"${cache_s2_actions:+--cache_s2_actions=${cache_s2_actions}} "

job_name="${data_name}_ck${s1_chunk_size}-${s2_chunk_size}-${vqh_chunk_size}_"\
"sh-${s1_his_state_size}_"\
"${theta1:+theta${theta1}-${theta2}_}"\
"${noise_slides_eps:+eps${noise_slides_eps}_}"\
"${noise_slides_alp:+alp${noise_slides_alp}_}"\
"${cache_s2_actions:+cache_}"\
"gpu${GPUS}_lr${lr}_${critic_lr}_${actor_lr}_${temp_lr}_"\
"bs${PER_DEVICE_BATCH_SIZE}_"\
"s$((steps / 1000))k"

pretrained_s2_path=${pretrained_map[$data_name]}
echo "pretrained_s2_path: ${pretrained_s2_path}"

# Launch training
echo "train_args: ${train_args}"
echo "job_name: ${job_name}"
accelerate launch $ACCELERATE_ARGS src/hume/training/train_vqh_s1.py ${train_args} \
  --pretrained_s2_path=${pretrained_s2_path} \
  --policy.type=hume \
  --pretrained_dino_path=/liujinxin/code/sj/dualAffordance/hume/checkpoints/dinov2-small \
  --config_path=config/${cfg}.json \
  --dataset.repo_id=${dataset} \
  --dataset.video_backend="pyav" \
  --dataset.image_transforms.enable=false \
  --dataset.rename_image_keys=true \
  --num_workers=${num_workers} \
  --policy_optimizer_lr=${lr} \
  --s1_chunk_size=${s1_chunk_size} \
  --s2_chunk_size=${s2_chunk_size} \
  --steps=${steps} \
  --batch=${PER_DEVICE_BATCH_SIZE} \
  --save_freq=$save_freq \
  --log_freq=100 \
  --job_name=${job_name} \
  --wandb.mode=offline \
  --wandb.enable=${wandb_enable} \
  --wandb.disable_artifact=true \
  --wandb.project=${WANDB_PROJECT} \
  --wandb.entity=${WANDB_ENTITY} \
  --next_obs_offset=${vqh_chunk_size} \
  --vqh_chunk_size=${vqh_chunk_size} \
  --critic_lr=${critic_lr} \
  --actor_lr=${actor_lr} \
  --temp_lr=${temp_lr} \
  --checkpoints_total_limit=0 \
  --s1_his_state_size=${s1_his_state_size}