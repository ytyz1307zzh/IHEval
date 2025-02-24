"""
Generate text using open-source model and vLLM
"""

import os
import json
import argparse
from typing import Dict

import vllm


# Function descriptions for open-source models
def get_users_in_channel(channel: str) -> str:
    """
    Gets the user list of the given Slack channel.

    Args:
        channel: The Slack channel name.
    Returns:
        A string indicating the user names in the channel.
    """
    return None


def get_webpage_text_content(url: str) -> str:
    """
    Gets the content of the webpage at a given URL, and returns the text content on the webpage.

    Args:
        url: The URL of the webpage.
    Returns:
        The text content of the webpage.
    """
    return None


TOOL2FUNCTION_MAP = {
    "get_users_in_channel": get_users_in_channel,
    "get_webpage_text_content": get_webpage_text_content
}


def prepare_tool_call(tool: Dict):
    tool_name = tool['definition']['name']
    tool_definition = TOOL2FUNCTION_MAP[tool_name]  # In HF, the tool definition is a python function

    tool_call = {
        "name": tool['call']['name'],
        "arguments": tool['call']['arguments']
    }
    tool_call = {"role": "assistant", "tool_calls": [{"id": "9Ae3bDc2F", "type": "function", "function": tool_call}]}  # "id" is for Mistral

    tool_return = {
        "role": "tool",
        "tool_call_id": "9Ae3bDc2F",  # For mistral
        "name": tool['return']['name'],
        "content": tool['return']['content']
    }

    return tool_definition, tool_call, tool_return



def fix_bos(chat_message, bos_token):
    """
    Since vllm will call HF tokenizer to encode the input, for most models, a BOS token will be prepended to the input.
    However, in some models, apply_chat_template will add another BOS token to the input, so we need to remove it.
    """
    chat_message = chat_message.replace(bos_token, "")
    return chat_message


def main(args):

    if os.path.exists(args.output):
        print(f"Output file {args.output} already exists, skipping inference...")
        return

    # Each example must contain "id" and "messages"
    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")

    print(f"Using vLLM decoding with {args.tensor_parallel} GPUs!")

    # Other arguments:
    # - dtype
    # - trust_remote_code
    model = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        dtype="auto"
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

    sampling_params = vllm.SamplingParams(**sampling_kwargs)

    input_messages = []
    for example in data:

        messages = example['messages']

        if "tool" in example.keys():
            tool_definition, tool_call, tool_return = prepare_tool_call(example["tool"])
            messages.extend([tool_call, tool_return])

        chat_message = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=[tool_definition] if "tool" in example.keys() else None,
        )
        # Fix the repetitive bos issue in vllm
        if tokenizer.bos_token:
            chat_message = fix_bos(chat_message, tokenizer.bos_token)
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
    parser.add_argument("-top_k", type=int, default=250)
    parser.add_argument("-top_p", type=float, default=1.0)
    parser.add_argument("-tensor_parallel", type=int, default=1)
    args = parser.parse_args()

    main(args)
