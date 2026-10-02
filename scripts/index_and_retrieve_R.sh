#!/bin/bash
set -e

# --------------------------------------------------------------------------------
# ------------ Set the following parameters with the correct values. -------------
# --------------------------------------------------------------------------------
FINETUNED_MODEL_OUTPUT_FOLDER="/path/to/root_folder/models/finetune_ModelName"
R_CORPUS_FILENAME="/path/to/dataset/SearchESCI/corpus.jsonl"

R_TRAIN_ESCI_QUERIES_FILENAME="/path/to/dataset/SearchESCI/train_queries.jsonl"
R_VALID_ESCI_QUERIES_FILENAME="/path/to/dataset/SearchESCI/valid_queries.jsonl"
R_TEST_ESCI_QUERIES_FILENAME="/path/to/dataset/SearchESCI/test_queries.jsonl"

R_INDEX_FOLDER="/path/to/root_folder/indexes/SearchESCI/finetune_ModelName"

R_TRAIN_ESCI_QUERY_INDEX_FOLDER="/path/to/root_folder/query_indexes/SearchESCI/train/finetune_ModelName"
R_VALID_ESCI_QUERY_INDEX_FOLDER="/path/to/root_folder/query_indexes/SearchESCI/valid/finetune_ModelName"
R_TEST_ESCI_QUERY_INDEX_FOLDER="/path/to/root_folder/query_indexes/SearchESCI/test/finetune_ModelName"

RUN_FOLDER="/path/to/root_folder/runs/finetune_ModelName"

# --------------------------------------------------------------------------------
# ------------------ Fixed parameters for the given model/task. ------------------
# --------------------------------------------------------------------------------
ID_FIELD="id"
QUERY_TEXT_FIELD="text"
R_TEXT_FIELD1="title"
R_TEXT_FIELD2="text"

CHUNK_SIZE=4096
BATCH_SIZE=64
MAX_NUM_TOKENS=128

RETRIEVAL_CHUNK_SIZE=10000
TOP_K=1000

Q_FINETUNED_MODEL="${FINETUNED_MODEL_OUTPUT_FOLDER}/q_model"
D_FINETUNED_MODEL="${FINETUNED_MODEL_OUTPUT_FOLDER}/d_model"

R_TRAIN_ESCI_RUN_FILENAME="${RUN_FOLDER}/r_train_esci.txt"
R_VALID_ESCI_RUN_FILENAME="${RUN_FOLDER}/r_valid_esci.txt"
R_TEST_ESCI_RUN_FILENAME="${RUN_FOLDER}/r_test_esci.txt"

# --------------------------------------------------------------------------------
# ----------------------- Check if the input files exists. -----------------------
# --------------------------------------------------------------------------------
if [[ ! -d "${FINETUNED_MODEL_OUTPUT_FOLDER}" ]]; then
  echo "Fine-tuned model input folder not found: ${FINETUNED_MODEL_OUTPUT_FOLDER}"
  exit 1
fi
if [[ ! -d "${Q_FINETUNED_MODEL}" ]]; then
  echo "Fine-tuned query model input folder not found: ${Q_FINETUNED_MODEL}"
  exit 1
fi
if [[ ! -f "${Q_FINETUNED_MODEL}/config.json" ]]; then
  echo "Fine-tuned query model input files inside the folder not found: ${Q_FINETUNED_MODEL}"
  exit 1
fi
if [[ ! -d "${D_FINETUNED_MODEL}" ]]; then
  echo "Fine-tuned document model input folder not found: ${D_FINETUNED_MODEL}"
  exit 1
fi
if [[ ! -f "${D_FINETUNED_MODEL}/config.json" ]]; then
  echo "Fine-tuned document model input files inside the folder not found: ${D_FINETUNED_MODEL}"
  exit 1
fi
if [[ ! -f "${R_CORPUS_FILENAME}" ]]; then
  echo "Training R corpus data input file not found: ${R_CORPUS_FILENAME}"
  exit 1
fi
if [[ ! -f "${R_TRAIN_ESCI_QUERIES_FILENAME}" ]]; then
  echo "Search ESCI training queries input file not found: ${R_TRAIN_ESCI_QUERIES_FILENAME}"
  exit 1
fi
if [[ ! -f "${R_VALID_ESCI_QUERIES_FILENAME}" ]]; then
  echo "Search ESCI validation queries input file not found: ${R_VALID_ESCI_QUERIES_FILENAME}"
  exit 1
fi
if [[ ! -f "${R_TEST_ESCI_QUERIES_FILENAME}" ]]; then
  echo "Search ESCI test queries input file not found: ${R_TEST_ESCI_QUERIES_FILENAME}"
  exit 1
fi

# --------------------------------------------------------------------------------
# ------------------- Create the output folders, if necessary. -------------------
# --------------------------------------------------------------------------------
if [[ ! -d "${R_INDEX_FOLDER}" ]]; then
  mkdir "${R_INDEX_FOLDER}";
fi
if [[ ! -d "${R_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output P index folder: ${R_INDEX_FOLDER}"
  exit 1
fi
if [[ ! -d "${R_TRAIN_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  mkdir "${R_TRAIN_ESCI_QUERY_INDEX_FOLDER}";
fi
if [[ ! -d "${R_TRAIN_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output Search ESCI training query index folder: ${R_TRAIN_ESCI_QUERY_INDEX_FOLDER}"
  exit 1
fi
if [[ ! -d "${R_VALID_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  mkdir "${R_VALID_ESCI_QUERY_INDEX_FOLDER}";
fi
if [[ ! -d "${R_VALID_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output Search ESCI validation query index folder: ${R_VALID_ESCI_QUERY_INDEX_FOLDER}"
  exit 1
fi
if [[ ! -d "${R_TEST_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  mkdir "${R_TEST_ESCI_QUERY_INDEX_FOLDER}";
fi
if [[ ! -d "${R_TEST_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output Search ESCI test query index folder: ${R_TEST_ESCI_QUERY_INDEX_FOLDER}"
  exit 1
fi
if [[ ! -d "${RUN_FOLDER}" ]]; then
  mkdir "${RUN_FOLDER}";
fi
if [[ ! -d "${RUN_FOLDER}" ]]; then
  echo "Unable to create the output run folder: ${RUN_FOLDER}"
  exit 1
fi

# --------------------------------------------------------------------------------
# ------------------------------ Indexing execution. -----------------------------
# --------------------------------------------------------------------------------
# Index the review corpus.
python "../src/indexing/indexing.py" \
  --corpus_filename="${R_CORPUS_FILENAME}" \
  --id_field="${ID_FIELD}" \
  --text_field1="${R_TEXT_FIELD1}" \
  --text_field2="${R_TEXT_FIELD2}" \
  --sort_by_length \
  --index_folder="${R_INDEX_FOLDER}" \
  --model="${D_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of review corpus completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Index the Search ESCI training queries.
python "../src/indexing/indexing.py" \
  --corpus_filename="${R_TEST_ESCI_QUERIES_FILENAME}" \
  --id_field="${ID_FIELD}" \
  --text_field1="${QUERY_TEXT_FIELD}" \
  --sort_by_length \
  --index_folder="${R_TEST_ESCI_QUERY_INDEX_FOLDER}" \
  --model="${Q_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of Search ESCI training queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Index the Search ESCI validation queries.
python "../src/indexing/indexing.py" \
  --corpus_filename="${R_VALID_ESCI_QUERIES_FILENAME}" \
  --id_field="${ID_FIELD}" \
  --text_field1="${QUERY_TEXT_FIELD}" \
  --sort_by_length \
  --index_folder="${R_VALID_ESCI_QUERY_INDEX_FOLDER}" \
  --model="${Q_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of Search ESCI validation queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Index the Search ESCI test queries.
python "../src/indexing/indexing.py" \
  --corpus_filename="${R_TEST_ESCI_QUERIES_FILENAME}" \
  --id_field="${ID_FIELD}" \
  --text_field1="${QUERY_TEXT_FIELD}" \
  --sort_by_length \
  --index_folder="${R_TEST_ESCI_QUERY_INDEX_FOLDER}" \
  --model="${Q_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of Search ESCI test queries completed."
echo "--------------------------------------------------"
echo ""
echo ""


# Perform retrieval on the Search ESCI training queries.
python "../src/indexing/retrieval.py" \
  --query_index_folder="${R_TRAIN_ESCI_QUERY_INDEX_FOLDER}" \
  --doc_index_folder="${R_INDEX_FOLDER}" \
  --run_filename="${R_TRAIN_ESCI_RUN_FILENAME}" \
  --top_k="${TOP_K}" \
  --chunk_size=${RETRIEVAL_CHUNK_SIZE}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Retrieval on Search ESCI training queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Perform retrieval on the Search ESCI validation queries.
python "../src/indexing/retrieval.py" \
  --query_index_folder="${R_VALID_ESCI_QUERY_INDEX_FOLDER}" \
  --doc_index_folder="${R_INDEX_FOLDER}" \
  --run_filename="${R_VALID_ESCI_RUN_FILENAME}" \
  --top_k="${TOP_K}" \
  --chunk_size=${RETRIEVAL_CHUNK_SIZE}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Retrieval on Search ESCI validation queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Perform retrieval on the Search ESCI test queries.
python "../src/indexing/retrieval.py" \
  --query_index_folder="${R_TEST_ESCI_QUERY_INDEX_FOLDER}" \
  --doc_index_folder="${R_INDEX_FOLDER}" \
  --run_filename="${R_TEST_ESCI_RUN_FILENAME}" \
  --top_k="${TOP_K}" \
  --chunk_size=${RETRIEVAL_CHUNK_SIZE}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Retrieval on Search ESCI test queries completed."
echo "--------------------------------------------------"
echo "--------------------------------------------------"
echo "Done Everything!"
echo ""
echo ""