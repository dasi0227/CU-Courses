# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The completed chain makes one vision request
per receipt and writes one aggregate amount for each query.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

```mermaid
flowchart LR
    A[Receipt images in folder] --> B[DeepSeek V4 Flash Vision via LangChain]
    B --> C[Extract per receipt: subtotal, final payment, discount lines]
    C --> D[Python Decimal aggregation]
    D --> E[Total paid after rounding]
    D --> F[Subtotal plus discounts, excluding rounding]
    E --> G[One HKD amount per exact query]
    F --> G
```

The chain sends each receipt image to `deepseek-v4-flash-vision-exp` through LangChain and asks for the printed subtotal, the final payment after rounding, and each discount line as structured JSON. The extraction prompt is stored in `prompt.md` and loaded relative to `hw1.py`. The runner adds final payments for the first answer; for the no-discount answer, it adds each receipt's discounts back to its subtotal while excluding rounding. Amounts are parsed and summed with `Decimal` to avoid floating-point rounding errors, and each response contains only one HKD amount.
