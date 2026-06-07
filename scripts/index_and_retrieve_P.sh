#!/bin/bash
set -e

# --------------------------------------------------------------------------------
# ------------ Set the following parameters with the correct values. -------------
# --------------------------------------------------------------------------------
FINETUNED_MODEL_OUTPUT_FOLDER="/path/to/root_folder/models/finetune_ModelName"
P_CORPUS_FILENAME="/path/to/dataset/AmazonESCI/catalogue.jsonl"

P_VALID_ESCI_QUERIES_FILENAME="/path/to/dataset/AmazonESCI/valid_queries.jsonl"
P_TEST_ESCI_QUERIES_FILENAME="/path/to/dataset/AmazonESCI/test_queries.jsonl"
P_TEST_PS23_QUERIES_FILENAME="/path/to/dataset/ProductSearch2023/test_queries.jsonl"

P_VALID_ESCI_QRELS_FILENAME="/path/to/dataset/AmazonESCI/valid_qrels.qrels"
P_TEST_ESCI_QRELS_FILENAME="/path/to/dataset/AmazonESCI/test_qrels.qrels"
P_TEST_PS23_QRELS_FILENAME="/path/to/dataset/ProductSearch2023/test_qrels.qrels"

P_INDEX_FOLDER="/path/to/root_folder/indexes/AmazonESCI/finetune_ModelName"
P_VALID_ESCI_QUERY_INDEX_FOLDER="/path/to/root_folder/query_indexes/AmazonESCI/finetune_ModelName"
P_TEST_ESCI_QUERY_INDEX_FOLDER="/path/to/root_folder/query_indexes/SearchESCI/finetune_ModelName"
P_TEST_PS23_QUERY_INDEX_FOLDER="/path/to/root_folder/query_indexes/ProductSearch2023/finetune_ModelName"
RUN_FOLDER="/path/to/root_folder/runs/finetune_ModelName"

# --------------------------------------------------------------------------------
# ------------------ Fixed parameters for the given model/task. ------------------
# --------------------------------------------------------------------------------
P_ID_FIELD="id"
P_TEXT_FIELD1="title"
P_TEXT_FIELD2="description"
P_TEXT_FIELD3="bullet_point"
P_QUERY_TEXT_FIELD="text"

CHUNK_SIZE=4096
BATCH_SIZE=64
MAX_NUM_TOKENS=128

RETRIEVAL_CHUNK_SIZE=10000
TOP_K=1000

Q_FINETUNED_MODEL="${FINETUNED_MODEL_OUTPUT_FOLDER}/q_model"
D_FINETUNED_MODEL="${FINETUNED_MODEL_OUTPUT_FOLDER}/d_model"

P_VALID_ESCI_RUN_FILENAME="${RUN_FOLDER}/p_valid_esci.txt"
P_TEST_ESCI_RUN_FILENAME="${RUN_FOLDER}/p_test_esci.txt"
P_TEST_PS23_RUN_FILENAME="${RUN_FOLDER}/p_test_ps23.txt"

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
if [[ ! -f "${P_CORPUS_FILENAME}" ]]; then
  echo "Training P corpus data input file not found: ${P_CORPUS_FILENAME}"
  exit 1
fi
if [[ ! -f "${P_VALID_ESCI_QUERIES_FILENAME}" ]]; then
  echo "Amazon ESCI validation queries input file not found: ${P_VALID_ESCI_QUERIES_FILENAME}"
  exit 1
fi
if [[ ! -f "${P_TEST_ESCI_QUERIES_FILENAME}" ]]; then
  echo "Amazon ESCI test queries input file not found: ${P_TEST_ESCI_QUERIES_FILENAME}"
  exit 1
fi
if [[ ! -f "${P_TEST_PS23_QUERIES_FILENAME}" ]]; then
  echo "TREC Product Search 2023 test queries input file not found: ${P_TEST_PS23_QUERIES_FILENAME}"
  exit 1
fi
if [[ ! -f "${P_VALID_ESCI_QRELS_FILENAME}" ]]; then
  echo "Amazon ESCI validation relevance judgements input file not found: ${P_VALID_ESCI_QRELS_FILENAME}"
  exit 1
fi
if [[ ! -f "${P_TEST_ESCI_QRELS_FILENAME}" ]]; then
  echo "Amazon ESCI test relevance judgements input file not found: ${P_TEST_ESCI_QRELS_FILENAME}"
  exit 1
fi
if [[ ! -f "${P_TEST_PS23_QRELS_FILENAME}" ]]; then
  echo "TREC Product Search 2023 test relevance judgements input file not found: ${P_TEST_PS23_QRELS_FILENAME}"
  exit 1
fi

# --------------------------------------------------------------------------------
# ------------------- Create the output folders, if necessary. -------------------
# --------------------------------------------------------------------------------
if [[ ! -d "${P_INDEX_FOLDER}" ]]; then
  mkdir "${P_INDEX_FOLDER}";
fi
if [[ ! -d "${P_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output P index folder: ${P_INDEX_FOLDER}"
  exit 1
fi
if [[ ! -d "${P_VALID_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  mkdir "${P_VALID_ESCI_QUERY_INDEX_FOLDER}";
fi
if [[ ! -d "${P_VALID_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output Amazon ESCI validation query index folder: ${P_VALID_ESCI_QUERY_INDEX_FOLDER}"
  exit 1
fi
if [[ ! -d "${P_TEST_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  mkdir "${P_TEST_ESCI_QUERY_INDEX_FOLDER}";
fi
if [[ ! -d "${P_TEST_ESCI_QUERY_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output Amazon ESCI test query index folder: ${P_TEST_ESCI_QUERY_INDEX_FOLDER}"
  exit 1
fi
if [[ ! -d "${P_TEST_PS23_QUERY_INDEX_FOLDER}" ]]; then
  mkdir "${P_TEST_PS23_QUERY_INDEX_FOLDER}";
fi
if [[ ! -d "${P_TEST_PS23_QUERY_INDEX_FOLDER}" ]]; then
  echo "Unable to create the output TREC Product Search 2023 test query index folder: ${P_TEST_PS23_QUERY_INDEX_FOLDER}"
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
# Index the product catalogue.
python "../src/indexing/indexing.py" \
  --corpus_filename="${P_CORPUS_FILENAME}" \
  --id_field="${P_ID_FIELD}" \
  --text_field1="${P_TEXT_FIELD1}" \
  --text_field2="${P_TEXT_FIELD2}" \
  --text_field3="${P_TEXT_FIELD3}" \
  --sort_by_length \
  --index_folder="${P_INDEX_FOLDER}" \
  --model="${D_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of product catalogue completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Index the Amazon ESCI validation queries.
python "../src/indexing/indexing.py" \
  --corpus_filename="${P_VALID_ESCI_QUERIES_FILENAME}" \
  --id_field="${P_ID_FIELD}" \
  --text_field1="${P_QUERY_TEXT_FIELD}" \
  --sort_by_length \
  --index_folder="${P_VALID_ESCI_QUERY_INDEX_FOLDER}" \
  --model="${Q_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of Amazon ESCI validation queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Index the Amazon ESCI test queries.
python "../src/indexing/indexing.py" \
  --corpus_filename="${P_TEST_ESCI_QUERIES_FILENAME}" \
  --id_field="${P_ID_FIELD}" \
  --text_field1="${P_QUERY_TEXT_FIELD}" \
  --sort_by_length \
  --index_folder="${P_TEST_ESCI_QUERY_INDEX_FOLDER}" \
  --model="${Q_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of Amazon ESCI test queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Index the TREC Product Search 2023 test queries.
python "../src/indexing/indexing.py" \
  --corpus_filename="${P_TEST_PS23_QUERIES_FILENAME}" \
  --id_field="${P_ID_FIELD}" \
  --text_field1="${P_QUERY_TEXT_FIELD}" \
  --sort_by_length \
  --index_folder="${P_TEST_PS23_QUERY_INDEX_FOLDER}" \
  --model="${Q_FINETUNED_MODEL}" \
  --custom_model \
  --chunk_size=${CHUNK_SIZE} \
  --batch_size=${BATCH_SIZE} \
  --max_num_tokens=${MAX_NUM_TOKENS}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Indexing of TREC Product Search 2023 test queries completed."
echo "--------------------------------------------------"
echo ""
echo ""


# Perform retrieval on the Amazon ESCI validation queries.
python "../src/indexing/retrieval.py" \
  --query_index_folder="${P_VALID_ESCI_QUERY_INDEX_FOLDER}" \
  --doc_index_folder="${P_INDEX_FOLDER}" \
  --run_filename="${P_VALID_ESCI_RUN_FILENAME}" \
  --top_k="${TOP_K}" \
  --chunk_size=${RETRIEVAL_CHUNK_SIZE}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Retrieval on Amazon ESCI validation queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Perform retrieval on the Amazon ESCI test queries.
python "../src/indexing/retrieval.py" \
  --query_index_folder="${P_TEST_ESCI_QUERY_INDEX_FOLDER}" \
  --doc_index_folder="${P_INDEX_FOLDER}" \
  --run_filename="${P_TEST_ESCI_RUN_FILENAME}" \
  --top_k="${TOP_K}" \
  --chunk_size=${RETRIEVAL_CHUNK_SIZE}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Retrieval on Amazon ESCI test queries completed."
echo "--------------------------------------------------"
echo ""
echo ""

# Perform retrieval on the TREC Product Search 2023 test queries.
python "../src/indexing/retrieval.py" \
  --query_index_folder="${P_TEST_PS23_QUERY_INDEX_FOLDER}" \
  --doc_index_folder="${P_INDEX_FOLDER}" \
  --run_filename="${P_TEST_PS23_RUN_FILENAME}" \
  --top_k="${TOP_K}" \
  --chunk_size=${RETRIEVAL_CHUNK_SIZE}
if [[ $? -ne 0 ]]; then
  exit 1;
fi

echo ""
echo "Retrieval on TREC Product Search 2023 test queries completed."
echo "--------------------------------------------------"
echo "--------------------------------------------------"
echo "Done Everything!"
echo ""
echo ""