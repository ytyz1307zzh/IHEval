import boto3
import json
import argparse
import os
import shutil
import datetime
import time
import pdb
from typing import Dict
from copy import deepcopy
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI
from functools import partial
from transformers import AutoTokenizer

# Mapping of the bedrock model name to HF model name
LLAMA_BEDROCK2HF_MAP = {
    "meta.llama3-8b-instruct-v1:0": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta.llama3-70b-instruct-v1:0": "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta.llama3-1-8b-instruct-v1:0": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta.llama3-1-70b-instruct-v1:0": "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

LLAMA_BEDROCK2HF_MAP_FOR_TOOL = {
    "meta.llama3-8b-instruct-v1:0": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta.llama3-70b-instruct-v1:0": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta.llama3-1-8b-instruct-v1:0": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta.llama3-1-70b-instruct-v1:0": "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

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


# OPENAI tool format: {'type': 'function', 'function': {'name': 'get_webpage', 'description': 'Returns the content of the webpage at a given URL.', 'parameters': {'properties': {'url': {'description': 'The URL of the webpage.', 'title': 'Url', 'type': 'string'}}, 'required': ['url'], 'title': 'Input schema for `get_webpage`', 'type': 'object'}}}
def tool_call_openai(tool: Dict):
    """
    Convert the tool call to OpenAI format.
    """
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    definition = [{
        'type': 'function',
        'function': {
            'name': raw_definition['name'],
            'description': raw_definition['description'],
            "parameters": {
                "type": "object",
                "properties": raw_definition['parameters'],
                "required": list(raw_definition['parameters'].keys()),
            }
        }
    }]

    tool_call = {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': raw_tool_call['id'],
                'type': 'function',
                'function': {
                    'name': raw_tool_call['name'],
                    'arguments': json.dumps(raw_tool_call['arguments'])
                }
            }
        ]
    }

    tool_return = {
        'role': 'tool',
        'tool_call_id': raw_tool_return['id'],
        'name': raw_tool_return['name'],
        'content': raw_tool_return['content']
    }

    return definition, tool_call, tool_return

# Claude tool format: [{'type': 'function', 'function': {'name': 'top_song', 'description': 'Get the most popular song played on a radio station.', 'input_schema': {'type': 'object', 'properties': {'sign': {'type': 'string', 'description': 'The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ and WKRP.'}}, 'required': ['sign']}}}]
# Llama tool format: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1


def tool_call_claude(tool: Dict):
    """
    Convert the tool call to Anthropic format.
    """
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    definition = [
        {
            "name": raw_definition["name"],
            "description": raw_definition["description"],
            "input_schema": {
                "type": "object",
                "properties": raw_definition["parameters"],
                "required": list(raw_definition['parameters'].keys())
            }
        }
    ]

    tool_call = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": raw_tool_call["id"].replace("call_", "toolu_"),
                "name": raw_tool_call["name"],
                "input": raw_tool_call["arguments"]
            }
        ],
    }

    tool_return = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": raw_tool_return["id"].replace("call_", "toolu_"),
                "content": raw_tool_return["content"]
            }
        ]
    }

    return definition, tool_call, tool_return


def tool_call_llama(tool: Dict):

    tool_name = tool['definition']['name']
    tool_definition = TOOL2FUNCTION_MAP[tool_name]  # In HF, the tool definition is a python function

    tool_call = {
        "name": tool['call']['name'],
        "arguments": tool['call']['arguments']
    }
    tool_call = {"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]}

    tool_return = {
        "role": "tool",
        "name": tool['return']['name'],
        "content": tool['return']['content']
    }

    return tool_definition, tool_call, tool_return



def convert_jsonl_to_json(file_path) -> None:
    """Convert a jsonl file to a json file."""
    shutil.copy(file_path, file_path + ".bak")
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.remove(file_path + ".bak")


def append_to_jsonl(data, filename: str) -> None:
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


def call_llama(l_data, tokenizer):
    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
    messages = deepcopy(l_data["messages"])

    if "system" in l_data.keys() and l_data["system"] is not None:
        messages.insert(0, {"role": "system", "content": l_data["system"]})

    if "tool" in l_data.keys():
        tool_definition, tool_call, tool_return = tool_call_llama(l_data["tool"])
        messages.extend([tool_call, tool_return])

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, tools=[tool_definition], add_generation_prompt=True)
    else:
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    body = json.dumps({
        "prompt": chat_prompt,
        "max_gen_len": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    })

    retries = 0
    while retries < args.max_retries:
        try:
            response = bedrock.invoke_model(body=body, modelId=args.model)
            response_body = json.loads(response.get("body").read())
            append_to_jsonl(
                {
                    "id": l_data["id"],
                    "input": messages,
                    "output": response_body,
                },
                args.output,
            )
            p_bar.update(1)
            break

        except Exception as e:
            retries += 1
            print(
                f"Error on call attempt {retries}: {e}"
            )  # If we've reached the maximum number of retries, record an error message (or handle as desired)
            if retries == args.max_retries:
                print(f"Retry exceed the max_retries {retries} times.")
                break
            time.sleep(args.sleep)


def call_claude(l_data):
    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    messages = l_data["messages"]
    if "tool" in l_data.keys():
        tool_definition, tool_call, tool_return = tool_call_claude(l_data["tool"])
        messages.extend([tool_call, tool_return])

    body = {
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "anthropic_version": "bedrock-2023-05-31",
    }

    if "system" in l_data.keys() and l_data["system"] is not None:
        body["system"] = l_data["system"]
    
    if "tool" in l_data.keys():
        body["tools"] = tool_definition

    body = json.dumps(body)

    retries = 0
    while retries < args.max_retries:
        try:
            response = bedrock.invoke_model(body=body, modelId=args.model)
            response_body = json.loads(response.get("body").read())
            append_to_jsonl(
                {
                    "id": l_data["id"],
                    "system": l_data["system"] if "system" in l_data.keys() else None,
                    "input": messages,
                    "output": response_body,
                },
                args.output,
            )
            p_bar.update(1)
            break

        except Exception as e:
            retries += 1
            print(
                f"Error on call attempt {retries}: {e}"
            )  # If we've reached the maximum number of retries, record an error message (or handle as desired)
            if retries == args.max_retries:
                print(f"Retry exceed the max_retries {retries} times.")
                break
            time.sleep(args.sleep)


def call_openai(l_data):

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = deepcopy(l_data["messages"])

    if "tool" in l_data.keys():
        tool_definition, tool_call, tool_return = tool_call_openai(l_data["tool"])
        messages.extend([tool_call, tool_return])

    if "system" in l_data.keys() and l_data["system"] is not None:
        messages.insert(0, {"role": "system", "content": l_data["system"]})

    retries = 0
    while retries < args.max_retries:

        try:
            chat_completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=tool_definition if "tool" in l_data.keys() else None,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                top_p=args.top_p,
            )
            response_body = chat_completion.choices[0].message.content
            append_to_jsonl(
                {
                    "id": l_data["id"],
                    "input": messages,
                    "output": response_body,
                },
                args.output,
            )
            p_bar.update(1)
            break

        except Exception as e:

            # For safety tasks, sometimes openai rejects the request due to repetitive patterns in the attack
            # we should identify such as a successful defense by outputting an empty string (which would be considered as a success)
            if isinstance(e, openai.BadRequestError) and "repetitive patterns in your prompt" in e.body['message']:
                append_to_jsonl(
                    {
                        "id": l_data["id"],
                        "input": messages,
                        "output": "",
                    },
                    args.output,
                )
                p_bar.update(1)
                break

            retries += 1
            print(
                f"Error on call attempt {retries}: {e}"
            )  # If we've reached the maximum number of retries, record an error message (or handle as desired)
            if retries == args.max_retries:
                print(f"Retry exceed the max_retries {retries} times.")
                break
            time.sleep(args.sleep)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default="")  # anthropic.claude-3-sonnet-20240229-v1:0
    parser.add_argument("-input", default=None)
    parser.add_argument("-output", default=None)
    parser.add_argument("-max_new_tokens", type=int, default=1024)
    parser.add_argument("-top_k", type=int, default=250)
    parser.add_argument("-top_p", type=float, default=0.999)
    parser.add_argument("-temperature", type=float, default=1.0)
    parser.add_argument("-max_loop", type=int, default=1)
    parser.add_argument("-max_retries", type=int, default=20)
    parser.add_argument("-max_threads", type=int, default=8)
    parser.add_argument("-sleep", type=int, default=2)
    args = parser.parse_args()

    # Input is a JSON file
    requests = json.load(open(args.input, "r", encoding="utf-8"))
    tstart = datetime.datetime.now()
    for _ in range(args.max_loop):

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
            save_list_as_jsonl(args.output, existing_data)  # convert back to JSONL, otherwise there will be errors

        dataset = []
        # Each data instance should have "id", "messages", and optionally "system"
        for l_data in requests:
            if l_data["id"] not in exist_ids:
                dataset.append(l_data)
        print("EXISTING DATA #: ", len(exist_ids))
        print("API Call REQUEST #: ", len(dataset))
        assert len(exist_ids) + len(dataset) == len(requests)

        if len(dataset) == 0:
            print("All data instances have been processed! Skipping...")
            break

        p_bar = tqdm(range(len(dataset)))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_threads
        ) as executor:
            if "anthropic" in args.model:
                executor.map(call_claude, dataset)

            elif "gpt" in args.model:
                executor.map(call_openai, dataset)

            elif "llama" in args.model:
                if 'tool' in requests[0].keys():
                    tokenizer = AutoTokenizer.from_pretrained(LLAMA_BEDROCK2HF_MAP_FOR_TOOL[args.model])
                else:
                    tokenizer = AutoTokenizer.from_pretrained(LLAMA_BEDROCK2HF_MAP[args.model])

                partial_call_llama = partial(call_llama, tokenizer=tokenizer)
                executor.map(partial_call_llama, dataset)

    tend = datetime.datetime.now()
    ttime = tend - tstart
    print("Total time for {} calls: {}".format(len(dataset), ttime))
    # Convert the final output to JSON
    convert_jsonl_to_json(args.output)
