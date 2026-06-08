import argparse
import generation
import io
import itertools
import json
import os
import random
import re
import time


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--r_corpus_filename",
                      dest="r_corpus_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_offsets_filename",
                      dest="r_offsets_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_products_filename",
                      dest="r_products_filename",
                      type=str,
                      required=True)
    args.add_argument("--output_filename",
                      dest="output_filename",
                      type=str,
                      required=True)
    args.add_argument("--llm",
                      dest="llm",
                      type=str,
                      default="Qwen/Qwen3.5-9B",
                      required=False)
    args.add_argument("--max_query_length",
                      dest="max_query_length",
                      type=int,
                      default=6,
                      required=False)
    args.add_argument("--num_generated_queries",
                      dest="num_generated_queries",
                      type=int,
                      default=100000,
                      required=False)

    return args.parse_args()


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Load the LLM from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    llm = generation.LLM(args.llm)

    print(f"LLM loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Define the prompt and additional parameters used for generation.
    # ------------------------------------------------------------------------------------------------------------------
    prompt_schema = [
        {
            "role": "system",
            "content": "You are a synthetic dataset generator. "
                       "Your task is to read a product review and generate one question-answer pair that a curious, "
                       "research-oriented user might ask when exploring a product category."
        },
        {
            "role": "user",
            "content": "Return only valid JSON in this exact format, with no preamble: "
                       "{\"question\": <question>, \"answer\": <answer>}\n"
                       "\n\n"
                       "---\n"
                       "### TARGET USER:\n"
                       "The user is researching products to inform a future purchase. They want objective, "
                       "factual information about product features, specifications, use cases, and performance. "
                       "They are not interested in the reviewer's personal opinions or subjective experiences.\n"
                       "\n\n"
                       "---\n"
                       "### INPUT REVIEW:\n"
                       "Title: #@review_title@#\n"
                       "Text: #@review_text@#\n"
                       "\n\n"
                       "---\n"
                       "### QUESTION RULES:\n"
                       "Generate exactly one question. It must satisfy ALL of the following:\n"
                       "1. **Fact-based**: Derived from objective facts in the document "
                       "(e.g., specs, features, use cases, compatibility). "
                       "Do not base the question on the reviewer's opinions, feelings, or personal experience.\n"
                       "2. **General**: Should make sense even if asked about a different brand or model "
                       "in the same product category. Avoid anchoring to the specific product name, model number, "
                       "or any detail that only applies to this exact item."
                       "3. **Self-contained**: A reader with no access to the document must fully understand it. "
                       "Do not reference \"the review\", \"the author\", \"the product above\", "
                       "or any implicit context.\n"
                       "4. **Answerable from the document alone**: Do not ask about anything that requires "
                       "knowledge beyond what the document provides.\n"
                       "5. **No embedded premises**: Do not include facts or assumptions in the question itself. "
                       "Ask openly (e.g., ask \"how does X work?\" not \"why is X better than Y?\").\n"
                       "6. **Objective**: Learning about a product or topic, not reading or summarizing a document. "
                       "The question must therefore feel natural in that context, "
                       "not like a reading comprehension exercise.\n"
                       "7. **Search and concise**: Phrased as a typed web query for search engines "
                       "(only keywords, without punctuation and without a natural-sounding structure). "
                       f"It consists of less than {args.max_query_length} words.\n"
                       "8. **Open-ended**: Prefer questions that invite elaboration over yes/no questions "
                       "(e.g., \"what are the main features of X?\" rather than \"does X have feature Y?\").\n"
                       "\n\n"
                       "---\n"
                       "### ANSWER RULES:\n"
                       "The answer must satisfy ALL of the following:\n"
                       "1. **Grounded**: Only include claims that are explicitly stated in the document "
                       "or are directly and unambiguously deducible from it. "
                       "Do not use world knowledge to fill gaps.\n"
                       "2. **Self-contained**: Do not reference \"the review\", \"the reviewer\", "
                       "or \"the document\". Write as if answering from general product knowledge.\n"
                       "3. **Objective**: Exclude the reviewer's opinions, subjective evaluations, "
                       "and personal anecdotes. Rephrase factual claims in neutral, third-person language.\n"
                       "4. **Faithful in terminology**: Use the same product names, technical terms, "
                       "and phrasing that appear in the document.\n"
                       "5. **User-agnostic**: Do not assume or mention anything about the user asking the question.\n"
                       "\n\n"
                       "---\n"
                       "### FALLBACK:\n"
                       "If the document contains no objective, verifiable facts "
                       "(e.g., it consists entirely of personal opinions with no factual claims), return:\n"
                       "{\"question\": null, \"answer\": null}\n"
                       "\n\n"
                       "---\n"
                       "### EXAMPLES:\n"
                       "BAD    → \"how often pee pad changed breeze litter system user habits\"\n"
                       "         ↳ references user personal habits, not a product fact\n"
                       "GOOD   → \"Logitech Harmony 1100 number of devices controlled simultaneously\"\n"
                       "         ↳ self-contained, but anchored to a specific SKU\n"
                       "BETTER → \"universal remote number of devices controlled simultaneously\"\n"
                       "         ↳ same specific fact, model name dropped\n"
                       "\n"
                       "BAD    → \"Star Wars Princess Leia Wig refund policy dissatisfaction\"\n"
                       "         ↳ asks about seller policy, not product facts\n"
                       "GOOD   → \"Ancestral Supplements Grass Fed Beef Liver capsules recommended daily dosage\"\n"
                       "         ↳ self-contained, but anchored to a specific supplement SKU\n"
                       "BETTER → \"beef liver supplement recommended daily dosage\"\n"
                       "         ↳ same specific fact, brand dropped, still answerable from the review\n"
                       "\n"
                       "BAD    → \"product main drawbacks problems\"\n"
                       "         ↳ generic template, not grounded in any fact from the review\n"
                       "GOOD   → \"King Arthur Better Cheddar Cheese Powder popcorn seasoning main ingredients\"\n"
                       "         ↳ brand and product line anchor to one specific SKU\n"
                       "BETTER → \"cheddar popcorn seasoning main ingredients\"\n"
                       "         ↳ brand removed, still fully answerable from the review\n"
        }
    ]

    gen_parameters = {
        "max_new_tokens": 1000,
        "pad_token_id": llm.tokenizer.eos_token_id
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Load the offset input data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.r_offsets_filename, "rt", encoding="utf-8") as fi:
        offsets_data = [*itertools.accumulate(map(int, fi))]
    del fi

    print(f"Load {len(offsets_data)} offsets in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Load the products input data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.r_products_filename, "rt", encoding="utf-8") as fi:
        products_data = {v["id"]: v["reviews_id"] for v in map(json.loads, fi)}
    del fi
    products_keys = sorted(products_data.keys())

    print(f"Load {len(products_keys)} product-review(s) mapping in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Process iteratively the input data.
    # ------------------------------------------------------------------------------------------------------------------
    list_discard = ["deliver", "mention", "packag", "product", "purchas", "refund", "review", "shipping",
                    "that", "these", "this", "those", "user", "warrant", "the book", "the movie", "the novel"]
    regex_discard = re.compile("the [^ ]+$")

    g_st = 0
    r_st = 0
    ctr = 0
    e_ctr = 0

    print("\n*** Start generating queries ***\n", flush=True)

    st = time.time_ns()
    with open(args.output_filename, "wt", encoding="utf-8") as fo, \
         open(args.r_corpus_filename, "rt", encoding="utf-8") as fi:
        while ctr < args.num_generated_queries:
            # Select a random review to use for generation.
            product_id = random.choice(products_keys)
            r_offset = random.choice([offsets_data[i] for i in products_data[product_id]])
            del product_id

            # Set the file to the correct byte offset.
            fi.seek(r_offset, io.SEEK_SET)
            del r_offset

            # Read the next review data from disk.
            r_st -= time.time_ns()
            data = json.loads(next(fi))
            r_st += time.time_ns()

            review_id = str(data["id"])
            product_id = str(data["asin"])
            review_title = str(data["title"])
            review_text = str(data["text"])
            del data

            prompt = [{
                "role": v1["role"],
                "content": v1["content"].replace("#@review_title@#", review_title)
                                        .replace("#@review_text@#", review_text)
            } for v1 in prompt_schema]

            g_st -= time.time_ns()
            llm_output = llm(prompt, **gen_parameters).strip()
            g_st += time.time_ns()
            del prompt

            try:
                llm_pair = json.loads(llm_output)
                llm_pair = {
                    # USEFUL data.
                    "query": llm_pair["question"].strip(),
                    "answer": llm_pair["answer"].strip(),
                    "review": review_id,
                    "asin": product_id
                }
            except:
                llm_pair = None
            del llm_output, review_title, review_text

            if llm_pair is None or "query" not in llm_pair.keys() or llm_pair["query"] is None or \
                    not isinstance(llm_pair["query"], str) or \
                    any(v in llm_pair["query"].lower() for v in list_discard) or \
                    regex_discard.search(llm_pair["query"]) is not None:
                llm_pair = None

            if llm_pair is not None:
                print(json.dumps(llm_pair), file=fo, flush=False)

                ctr += 1
                if (ctr % 100) == 0:
                    fo.flush()

                    print(f"Generated {ctr} queries in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=False)
                    print(f"  Reading time: {r_st / NS_IN_S:.3f} s.", flush=False)
                    print(f"  Generation time: {g_st / NS_IN_S:.3f} s.", flush=False)
                    print(f"  Errors in generation: {e_ctr}.\n", flush=True)
            else:
                e_ctr += 1
            del llm_pair

        fo.flush()
    del fo, fi

    if (ctr % 100) != 0:
        print(f"Generated {ctr} queries in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=False)
        print(f"  Reading time: {r_st / NS_IN_S:.3f} s.", flush=False)
        print(f"  Generation time: {g_st / NS_IN_S:.3f} s.", flush=False)
        print(f"  Errors in generation: {e_ctr}.\n", flush=True)
    del ctr, e_ctr, r_st, g_st, st

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
