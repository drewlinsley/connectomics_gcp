
mode=train_and_eval

enable_lars=False
train_batch_size=256
base_learning_rate=0.1
use_tpu=True

experiment_name=Test  # $1  # finetune_BU_{bu_loss}_TD_{td_loss}_R50_lr0.1_T0.1
# tpu_name=$2

# export TPU_NAME=$tpu_name  # 'prj-selfsup-tpu'
export STORAGE_BUCKET='gs://serrelab'
# DATA_DIR=$STORAGE_BUCKET/imagenet_dataset/imagenet2012/5.0.0/
DATA_BUCKET=$STORAGE_BUCKET/connectomics
MODEL_DIR=$DATA_BUCKET/connectomics/results/$experiment_name
gsutil mkdir $MODEL_DIR

TRAIN_FILES="${DATA_BUCKET}/tfrecords/celltype/*train*.tfrecords"
VAL_FILES="${DATA_BUCKET}/tfrecords/celltype/*val*.tfrecords"
TS=$(date +%s)
EXP_NAME=$experiment_name_${TS}
# EXPORT_DIR=$DATA_BUCKET/connectomics/served_models/${EXP_NAME}

bash get_ip.sh

python3 src/models/unet3d/unet_main.py \
--use_tpu \
--model_dir=$MODEL_DIR \
--training_file_pattern="${TRAIN_FILES}" \
--eval_file_pattern="${VAL_FILES}" \
--iterations_per_loop=10 \
--mode=train \
--config_file="src/models/unet3d/configs/cloud/v3-8_128x128x128_ce.yaml"
# --export_dir=$EXPORT_DIR


# --params_override="{\"optimizer\":\"momentum\",\"train_steps\":100}" \
