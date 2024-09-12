"""
Generate text using open-source model via the original interface of Huggingface
"""

import os
import json
import argparse

import torch
import transformers

PRECISION_MAP = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}

LLAMA3_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]


def main(args):

    # Each example must contain "id" and "messages"
    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        torch_dtype="auto" if args.precision == "auto" else PRECISION_MAP[args.precision],
        device_map="auto",
    )

    # For llama-3, add EOT as a termination token
    if args.model in LLAMA3_MODELS:
        stop_token_ids = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    input_messages = []
    for example in data:
        input_messages.append(example['messages'])

    print("*" * 15 + " Example input " + "*" * 15)
    print(input_messages[0])
    print("*" * 45)

    if args.temperature > 0:
        generation_kwargs = {
            "do_sample": True,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
    else:
        generation_kwargs = {
            "do_sample": False,
        }

    outputs = pipeline(
        input_messages,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=stop_token_ids,
        return_full_text=False,
        **generation_kwargs
    )

    results = []
    for i in range(len(outputs)):
        if i == 0:
            print(outputs[i])

        output = outputs[i][0]["generated_text"]
        results.append({
            "id": data[i]['id'],
            "input": input_messages[i],
            "generator": args.model,
            "output": output,
        })
    
    results = sorted(results, key=lambda x: x["id"])
    json.dump(
        results,
        open(args.output, "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )
    print(f"Saved {len(results)} examples to {args.output}.")

    print("*" * 15 + " Example output " + "*" * 15)
    print(results[0])
    print("*" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="gpt2")
    parser.add_argument("-input", type=str, default="data/input_data.json")
    parser.add_argument("-output", type=str, default="data/output_data.json")
    parser.add_argument("-max_new_tokens", type=int, default=100)
    parser.add_argument("-temperature", type=float, default=0.0)
    parser.add_argument("-top_k", type=float, default=250)
    parser.add_argument("-top_p", type=float, default=1.0)
    parser.add_argument("-precision", type=str, default="auto")
    args = parser.parse_args()

    main(args)
