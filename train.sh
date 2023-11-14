DATA_PATH="data/wmt19_v18"
TRAIN_PATH="./data/train_ready.txt"
VALID_PATH="./data/valid_ready.txt"

MODEL_NAME_OR_PATH="ai-forever/mGPT"

GPU_ID=0
LENGTH=64
BATCH_SIZE=1
LR="4e-5"
N_EPOCHS=500
RANDOM_SEED=2023

TRAIN_SIZE=50000
VALID_SIZE=10000
SHUFFLE_RATIO=0.2

# * ========== DO THIS ONE TIME ONLY ==========
python preprocess.py \
    --data_path ${DATA_PATH} \
    --train_file ${TRAIN_PATH} \
    --valid_file ${VALID_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --max_length ${LENGTH} \
    --seed ${RANDOM_SEED} \
    --train_size ${TRAIN_SIZE} \
    --valid_size ${VALID_SIZE} \
    --shuffle_ratio ${SHUFFLE_RATIO}
# * ==================================================
# TYPE="train"
# python train_adv.py \
#     --train_data_path ${TRAIN_PATH}\
#     --valid_data_path ${VALID_PATH} \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --max_length ${LENGTH} \
#     --batch_size ${BATCH_SIZE} \
#     --eval_batch_size ${BATCH_SIZE} \
#     --num_epochs ${N_EPOCHS} \
#     --learning_rate ${LR} \
#     --train_size ${TRAIN_SIZE} \
#     --seed ${RANDOM_SEED} \
#     --type ${TYPE}