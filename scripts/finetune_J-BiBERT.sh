#!/bin/bash
set -e

# --------------------------------------------------------------------------------
# ------------ Set the following parameters with the correct values. -------------
# --------------------------------------------------------------------------------
PRETRAINED_MODEL_OUTPUT_FOLDER="/path/to/root_folder/models/pretrain_J-BiBERT"
FINETUNED_MODEL_OUTPUT_FOLDER="/path/to/root_folder/models/finetune_J-BiBERT"
P_CORPUS_FILENAME="/path/to/dataset/AmazonESCI/catalogue.jsonl"
R_CORPUS_FILENAME="/path/to/dataset/SearchESCI/corpus.jsonl"
P_QUERIES_FILENAME="/path/to/dataset/AmazonESCI/train_queries.jsonl"
R_QUERIES_FILENAME="/path/to/dataset/SearchESCI/train_queries.jsonl"
P_TRAIN_FILENAME="/path/to/train-data_dataset/p_train_data.jsonl"
R_TRAIN_FILENAME="/path/to/train-data_dataset/r_train_data.jsonl"

NUM_TRAIN_EPOCHS=20
LEARNING_RATE=5e-6
BATCH_SIZE=64
MAX_NUM_TOKENS=128
GRADIENT_ACCUMULATION_STEPS=1
FINETUNE_ALPHA=0.0

# --------------------------------------------------------------------------------
# -------------------- Fixed parameters for the given model. ---------------------
# --------------------------------------------------------------------------------
TASK="joint"
NUM_NEGATIVES=7

LOSS_LOGGING_FILENAME="${FINETUNED_MODEL_OUTPUT_FOLDER}/loss_logging.tsv"
Q_MODEL_OUTPUT_FOLDER="${FINETUNED_MODEL_OUTPUT_FOLDER}/q_model"
D_MODEL_OUTPUT_FOLDER="${FINETUNED_MODEL_OUTPUT_FOLDER}/d_model"
T_MODEL_OUTPUT_FOLDER="${FINETUNED_MODEL_OUTPUT_FOLDER}/train_model"

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
if [[ ! -f "${P_TRAIN_FILENAME}" ]]; then
  echo "Training P train data input file not found: ${P_TRAIN_FILENAME}"
  exit 1
fi
if [[ ! -f "${R_TRAIN_FILENAME}" ]]; then
  echo "Training R train data input file not found: ${R_TRAIN_FILENAME}"
  exit 1
fi
if [[ ! -d "${PRETRAINED_MODEL_OUTPUT_FOLDER}" ]]; then
  echo "Pre-trained model input folder not found: ${PRETRAINED_MODEL_OUTPUT_FOLDER}"
  exit 1
fi
if [[ ! -f "${PRETRAINED_MODEL_OUTPUT_FOLDER}/config.json" ]]; then
  echo "Pre-trained model input files inside the folder not found: ${PRETRAINED_MODEL_OUTPUT_FOLDER}"
  exit 1
fi

# --------------------------------------------------------------------------------
# ------------------- Create the output folders, if necessary. -------------------
# --------------------------------------------------------------------------------
if [[ ! -d "${FINETUNED_MODEL_OUTPUT_FOLDER}" ]]; then
  mkdir "${FINETUNED_MODEL_OUTPUT_FOLDER}";
fi
if [[ ! -d "${FINETUNED_MODEL_OUTPUT_FOLDER}" ]]; then
  echo "Unable to create the output model folder: ${FINETUNED_MODEL_OUTPUT_FOLDER}"
  exit 1
fi
if [[ ! -d "${Q_MODEL_OUTPUT_FOLDER}" ]]; then
  mkdir "${Q_MODEL_OUTPUT_FOLDER}";
fi
if [[ ! -d "${Q_MODEL_OUTPUT_FOLDER}" ]]; then
  echo "Unable to create the output query model folder: ${Q_MODEL_OUTPUT_FOLDER}"
  exit 1
fi
if [[ ! -d "${D_MODEL_OUTPUT_FOLDER}" ]]; then
  mkdir "${D_MODEL_OUTPUT_FOLDER}";
fi
if [[ ! -d "${D_MODEL_OUTPUT_FOLDER}" ]]; then
  echo "Unable to create the output document model folder: ${D_MODEL_OUTPUT_FOLDER}"
  exit 1
fi
if [[ ! -d "${T_MODEL_OUTPUT_FOLDER}" ]]; then
  mkdir "${T_MODEL_OUTPUT_FOLDER}";
fi
if [[ ! -d "${T_MODEL_OUTPUT_FOLDER}" ]]; then
  echo "Unable to create the output train model folder: ${T_MODEL_OUTPUT_FOLDER}"
  exit 1
fi

# --------------------------------------------------------------------------------
# ------------------------------ Training execution. -----------------------------
# --------------------------------------------------------------------------------
python "../src/training/finetune.py" \
  --base_model="${PRETRAINED_MODEL_OUTPUT_FOLDER}" \
  --query_model_folder="${Q_MODEL_OUTPUT_FOLDER}" \
  --document_model_folder="${D_MODEL_OUTPUT_FOLDER}" \
  --train_model_folder="${T_MODEL_OUTPUT_FOLDER}" \
  --loss_logging_filename="${LOSS_LOGGING_FILENAME}" \
  --task="${TASK}" \
  --p_corpus_filename="${P_CORPUS_FILENAME}" \
  --r_corpus_filename="${R_CORPUS_FILENAME}" \
  --p_queries_filename="${P_QUERIES_FILENAME}" \
  --r_queries_filename="${R_QUERIES_FILENAME}" \
  --p_train_filename="${P_TRAIN_FILENAME}" \
  --r_train_filename="${R_TRAIN_FILENAME}" \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --learning_rate=${LEARNING_RATE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --no_gradient_checkpointing \
  --tie_qd_models \
  --finetune_alpha=${FINETUNE_ALPHA} \
  --num_positives=1 \
  --num_negatives=${NUM_NEGATIVES}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "--------------------------------------------------"
echo "Done Everything!"
echo ""
echo ""
