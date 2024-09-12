"""
Generate text using open-source model and vLLM
"""

import os
import json
import argparse

import vllm
import torch

PRECISION_MAP = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}

LLAMA3_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]

LLAMA2_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf"
]


def fix_bos(chat_message, model):
    """
    Since vllm will call HF tokenizer to encode the input, for most models, a BOS token will be prepended to the input.
    However, in some models, apply_chat_template will add another BOS token to the input, so we need to remove it.
    """
    if model in LLAMA3_MODELS:
        chat_message = chat_message.replace("<|begin_of_text|>", "")
    elif model in LLAMA2_MODELS:
        chat_message = chat_message.replace("<s>", "")
    else:
        pass

    return chat_message


def main(args):

    # Each example must contain "id" and "messages"
    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")

    print(f"Using vLLM decoding with {torch.cuda.device_count()} GPUs!")

    # Other arguments:
    # - dtype
    # - trust_remote_code
    model = vllm.LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        dtype="auto" if args.precision == "auto" else PRECISION_MAP[args.precision],
    )
    tokenizer = model.get_tokenizer()

    # Other arguments:
    # - repetition_penalty
    sampling_kwargs = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }

    # For llama-3, add EOT as a termination token
    if args.model in LLAMA3_MODELS:
        stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        sampling_kwargs["stop_token_ids"] = stop_token_ids

    sampling_params = vllm.SamplingParams(**sampling_kwargs)

    input_messages = []
    for example in data:
        chat_message = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        chat_message = fix_bos(chat_message, args.model)
        input_messages.append(chat_message)

    print("*" * 15 + " Example input " + "*" * 15)
    print(input_messages[0])
    print("*" * 45)

    outputs = model.generate(input_messages, sampling_params)

    results = []
    for i in range(len(data)):
        results.append(
            {
                "id": data[i]["id"],
                "input": outputs[i].prompt,
                "generator": args.model,
                "output": outputs[i].outputs[0].text,
            }
        )

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
