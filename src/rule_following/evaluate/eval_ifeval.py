"""
Evaluate the IFEval task
"""
from typing import Dict, List
import json
import os

import colorama
from termcolor import colored
colorama.init()


def save_list_as_jsonl(path: str, data):
    assert path.endswith(".jsonl")
    with open(path, "w", encoding="utf8") as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write("\n")


def eval_ifeval(args, id2answer: Dict, responses: List, extract_output: callable):
    """
    Format the data according to the format required by IFEval's official evaluation script, then call their official evaluation script
    """
    input_data = json.load(open(args.input, "r", encoding="utf-8"))
    id2instruction = {example["id"]: example["instruction"] for example in input_data}

    results = []
    for example in responses:
        id_ = example["id"]
        instruction = id2instruction[id_]
        prediction = extract_output(args, example)
        results.append({"prompt": instruction, "response": prediction})

    response_save_path = os.path.join(args.eval_output_dir, "ifeval_responses.jsonl")
    save_list_as_jsonl(response_save_path, results)

    evaluation_command = (
        f"python src/rule_following/evaluate/evaluation_main.py"
        f" --input_data={args.input}"
        f" --input_response_data={response_save_path}"
        f" --output_dir={args.eval_output_dir}"
    )

    print(colored("-" * 20 + "Evaluation" + "-" * 20, "green"))
    print(evaluation_command)
    os.system(evaluation_command)
    