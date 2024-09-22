"""
Generate text using open-source model and ollama
"""

import os
import pdb
import json
import shutil
import ollama
import requests
import argparse
import concurrent
from tqdm import tqdm
from typing import Dict
from copy import deepcopy
from transformers import AutoTokenizer


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


def append_to_jsonl(data, filename: str):
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def save_list_as_jsonl(path: str, data):
    with open(path, "w", encoding="utf8") as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write("\n")
    print(f"Saved {len(data)} data to {path}")


def convert_jsonl_to_json(file_path) -> None:
    """Convert a jsonl file to a json file."""
    shutil.copy(file_path, file_path + ".bak")
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.remove(file_path + ".bak")


def tool_call_mistral_7b(tool: Dict):
    tool_name = tool['definition']['name']
    tool_definition = TOOL2FUNCTION_MAP[tool_name]  # In HF, the tool definition is a python function

    tool_call = {
        "name": tool['call']['name'],
        "arguments": tool['call']['arguments']
    }
    tool_call = {"role": "assistant", "tool_calls": [{"id": "9Ae3bDc2F", "type": "function", "function": tool_call}]}

    tool_return = {
        "role": "tool",
        "tool_call_id": "9Ae3bDc2F",
        "name": tool['return']['name'],
        "content": tool['return']['content']
    }

    return tool_definition, tool_call, tool_return


def tool_call_ollama(tool: Dict):
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    tool_definition = {
        "type": "function",
        "function": {
            "name": raw_definition["name"],
            "description": raw_definition["description"],
            "parameters": {
                "type": "object",
                "properties": raw_definition["parameters"],
                "required": list(raw_definition["parameters"].keys()),
            },
        },
    }

    tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": raw_tool_call["name"],
                "arguments": raw_tool_call["arguments"],
            }
        }],
    }

    tool_return = {
        "role": "tool",
        "name": raw_tool_return["name"],
        "content": raw_tool_return["content"],
    }

    return tool_definition, tool_call, tool_return


def call_mistral(l_data):

    messages = deepcopy(l_data["messages"])

    if "tool" in l_data.keys():
        tool_definition, tool_call, tool_return = tool_call_mistral_7b(l_data["tool"])
        messages.extend([tool_call, tool_return])

    messages = tokenizer.apply_chat_template(
        messages,
        tools=[tool_definition] if "tool" in l_data.keys() else None,
        tokenize=False,
        add_generation_prompt=True,
    )

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": args.model,
        "prompt": messages,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": args.temperature,
            "num_predict": args.max_new_tokens,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response_body = response.json()['response'].strip()
    
    append_to_jsonl(
        {
            "id": l_data["id"],
            "input": messages,
            "output": response_body,
        },
        args.output,
    )
    p_bar.update(1)


def call_ollama(l_data):

    messages = deepcopy(l_data["messages"])

    if "tool" in l_data.keys():
        tool_definition, tool_call, tool_return = tool_call_ollama(l_data["tool"])
        messages.extend([tool_call, tool_return])

    response = ollama.chat(
        model=args.model,
        messages=messages,
        tools=[tool_definition] if "tool" in l_data.keys() else None,
        options={
            "temperature": args.temperature,
            "num_predict": args.max_new_tokens,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
    )

    response_body = response["message"]["content"].strip()

    append_to_jsonl(
        {
            "id": l_data["id"],
            "input": messages,
            "output": response_body,
        },
        args.output,
    )
    p_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="gpt2")
    parser.add_argument("-input", type=str, default="data/input_data.json")
    parser.add_argument("-output", type=str, default="data/output_data.json")
    parser.add_argument("-max_new_tokens", type=int, default=100)
    parser.add_argument("-temperature", type=float, default=0.0)
    parser.add_argument("-top_k", type=float, default=250)
    parser.add_argument("-top_p", type=float, default=1.0)
    parser.add_argument("-max_threads", type=int, default=1)
    args = parser.parse_args()

    try:
        exist_ids = set()
        if os.path.isfile(args.output):
            with open(args.output) as f:
                for line in f.readlines():
                    l_data = json.loads(line)
                    exist_ids.add(l_data["id"])

    # If inference has finished and the output has been converted to JSON, this will cause an error
    except json.decoder.JSONDecodeError:
        existing_data = json.load(open(args.output, "r", encoding="utf-8"))
        exist_ids = {l_data["id"] for l_data in existing_data}
        save_list_as_jsonl(
            args.output, existing_data
        )  # convert back to JSONL, otherwise there will be errors

    # Each example must contain "id" and "messages"
    all_requests = json.load(open(args.input, "r", encoding="utf-8"))
    dataset = []

    # Each data instance should have "id" and "messages"
    for l_data in all_requests:
        if l_data["id"] not in exist_ids:
            dataset.append(l_data)
    print("EXISTING DATA #: ", len(exist_ids))
    print("API Call REQUEST #: ", len(dataset))
    assert len(exist_ids) + len(dataset) == len(all_requests)

    if args.model.startswith("mistral:7b"):
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    p_bar = tqdm(range(len(dataset)))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_threads
    ) as executor:
        if args.model.startswith("mistral:7b"):
            executor.map(call_mistral, dataset)
        else:
            executor.map(call_ollama, dataset)

    convert_jsonl_to_json(args.output)
