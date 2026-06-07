#!/bin/bash
set -e

# --------------------------------------------------------------------------------
# ------------ Set the following parameters with the correct values. -------------
# --------------------------------------------------------------------------------
INIT_MODEL_OUTPUT_FOLDER="/path/to/root_folder/models/init_R-BiBERT"
PRETRAIN_MODEL_OUTPUT_FOLDER="/path/to/root_folder/models/pretrain_R-BiBERT"
R_CORPUS_FILENAME="/path/to/SearchESCI_dataset/corpus.jsonl"
R_QUERIES_FILENAME="/path/to/SearchESCI_dataset/train_queries.jsonl"

NUM_TRAIN_EPOCHS=20
LEARNING_RATE=1e-4
BATCH_SIZE=64
MAX_NUM_TOKENS=128
GRADIENT_ACCUMULATION_STEPS=1
PRETRAIN_ALPHA=0.10

# --------------------------------------------------------------------------------
# -------------------- Fixed parameters for the given model. ---------------------
# --------------------------------------------------------------------------------
BASE_MODEL="bert"
ARCHITECTURE="bibert"
TASK="review"
MLM_PROBABILITY=0.15
MASK_REPLACE_PROBABILITY=0.80
RANDOM_REPLACE_PROBABILITY=0.10

P_CORPUS_FILENAME="/path/to/empty.jsonl"
P_QUERIES_FILENAME="/path/to/empty.jsonl"
LOSS_LOGGING_FILENAME="${PRETRAIN_MODEL_OUTPUT_FOLDER}/loss_logging.tsv"

# --------------------------------------------------------------------------------
# ----------------------- Check if the input files exists. -----------------------
# --------------------------------------------------------------------------------
if [[ ! -f "${P_CORPUS_FILENAME}" ]]; then
  echo "Training P corpus data input file not found: ${P_CORPUS_FILENAME}"
  exit 1
fi
if [[ ! -f "${R_CORPUS_FILENAME}" ]]; then
  echo "Training R corpus data input file not found: ${R_CORPUS_FILENAME}"
  exit 1
fi
if [[ ! -f "${P_QUERIES_FILENAME}" ]]; then
  echo "Training P queries data input file not found: ${P_QUERIES_FILENAME}"
  exit 1
fi
if [[ ! -f "${R_QUERIES_FILENAME}" ]]; then
  echo "Training R queries data input file not found: ${R_QUERIES_FILENAME}"
  exit 1
fi

# --------------------------------------------------------------------------------
# ------------------- Create the output folders, if necessary. -------------------
# --------------------------------------------------------------------------------
if [[ ! -d "${INIT_MODEL_OUTPUT_FOLDER}" ]]; then
  mkdir "${INIT_MODEL_OUTPUT_FOLDER}";
fi
if [[ ! -d "${PRETRAIN_MODEL_OUTPUT_FOLDER}" ]]; then
  mkdir "${PRETRAIN_MODEL_OUTPUT_FOLDER}";
fi

# --------------------------------------------------------------------------------
# ------------------------------ Training execution. -----------------------------
# --------------------------------------------------------------------------------
python "../src/training/init.py" \
  --base_model="${BASE_MODEL}" \
  --architecture="${ARCHITECTURE}" \
  --task="${TASK}" \
  --output_folder="${INIT_MODEL_OUTPUT_FOLDER}"
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "--------------------------------------------------"
echo ""

python "../src/training/pretrain.py" \
  --base_model="${INIT_MODEL_OUTPUT_FOLDER}" \
  --output_folder="${PRETRAIN_MODEL_OUTPUT_FOLDER}" \
  --loss_logging_filename="${LOSS_LOGGING_FILENAME}" \
  --task="${TASK}" \
  --p_corpus_filename="${P_CORPUS_FILENAME}" \
  --r_corpus_filename="${R_CORPUS_FILENAME}" \
  --p_queries_filename="${P_QUERIES_FILENAME}" \
  --r_queries_filename="${R_QUERIES_FILENAME}" \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --learning_rate=${LEARNING_RATE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --no_gradient_checkpointing \
  --mlm_probability=${MLM_PROBABILITY} \
  --mask_replace_probability=${MASK_REPLACE_PROBABILITY} \
  --random_replace_probability=${RANDOM_REPLACE_PROBABILITY} \
  --pretrain_alpha=${PRETRAIN_ALPHA}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "--------------------------------------------------"
echo "Done Everything!"
echo ""
echo ""
