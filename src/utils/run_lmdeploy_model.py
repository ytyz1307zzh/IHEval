"""
Generate text using open-source model and vLLM
"""

import os
import json
import time
import argparse

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import torch


def main(args):

    if os.path.exists(args.output):
        print(f"Output file {args.output} already exists, skipping inference...")
        return

    # Each example must contain "id" and "messages"
    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")

    print(f"Using LMDeploy decoding with {torch.cuda.device_count()} GPUs!")

    backend_config = TurbomindEngineConfig(tp=torch.cuda.device_count())
    generation_config = GenerationConfig(
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    pipe = pipeline(args.model, backend_config=backend_config, log_level="WARNING")

    input_messages = [x["messages"] for x in data]

    print("*" * 15 + " Example input " + "*" * 15)
    print(input_messages[0])
    print("*" * 45)

    print("Start decoding!")
    start_time = time.time()
    outputs = pipe(input_messages, gen_config=generation_config)
    print(f'Elapsed time: {time.time() - start_time:.2f}s')

    results = []
    for i in range(len(data)):
        results.append(
            {
                "id": data[i]["id"],
                "input": data[i]["messages"],
                "generator": args.model,
                "output": outputs[i].text.strip(),
            }
        )

    # If there are non-integer ids, sorting may fail
    try:
        results = sorted(results, key=lambda x: x["id"])
    except Exception as e:
        print(f'Skip sorting results due to error: {e}')
        
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
