# Search ESCI

Search ESCI [https://huggingface.co/datasets/J-MADRAL/SearchESCI](https://huggingface.co/datasets/J-MADRAL/SearchESCI) is a novel synthetic large-scale test collection for retrieval search. It comprises a corpus of 22.1M reviews and a LLM-generated set of 75k, 12.5k, and 12.5k queries in the training, validation, and test sets, respectively. A single positive review is associated with each query, similarly to MS-MARCO dataset.

## Generation Settings ##

We employed [Qwen 3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B) open-source large language model to generate the set of queries, using the following settings:
- `all default generation settings`
* `max_new_tokens`: 1000
* `thinking`: disabled (to significantly reduce runtime, with little to no quality degradation)

## Prompt Used ##

```
{
    "role": "system",
    "content": "You are a synthetic dataset generator. "
               "Your task is to read a product review and generate one question-answer pair that a curious, "
               "research-oriented user might ask when exploring a product category."
}
```

```
{
    "role": "user",
    "content": "Return only valid JSON in this exact format, with no preamble: "
               "{\"question\": <question>, \"answer\": <answer>}\n"
               "\n"
               "\n"
               "---\n"
               "### TARGET USER:\n"
               "The user is researching products to inform a future purchase. They want objective, "
               "factual information about product features, specifications, use cases, and performance. "
               "They are not interested in the reviewer's personal opinions or subjective experiences.\n"
               "\n"
               "\n"
               "---\n"
               "### INPUT REVIEW:\n"
               "Title: #@review_title@#\n"
               "Text: #@review_text@#\n"
               "\n"
               "\n"
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
               "\n"
               "\n"
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
               "\n"
               "\n"
               "---\n"
               "### FALLBACK:\n"
               "If the document contains no objective, verifiable facts "
               "(e.g., it consists entirely of personal opinions with no factual claims), return:\n"
               "{\"question\": null, \"answer\": null}\n"
               "\n"
               "\n"
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
```

## Rule-based Filtering

We discarded any generated query containing any of the strings belonging to the following categories:

- `shopping-related`:
    * `deliver`, `packag`, `product`, `purchas`, `review`, `shipping`, `warrant`, `user`
- `stopwords`:
    * `mention`, `these`, `this`
- `common generic words`:
    * `the book`, `the movie`, `the novel`
