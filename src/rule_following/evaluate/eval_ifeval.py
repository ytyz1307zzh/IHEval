"""
Evaluate the IFEval task
"""
from typing import Dict, List
import json
import os


def save_list_as_jsonl(path: str, data):
    assert path.endswith(".jsonl")
    with open(path, "w", encoding="utf8") as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write("\n")
    print(f"Saved {len(data)} data to {path}")


def eval_ifeval(args, id2answer: Dict, responses: List, extract_output: callable):
    """
    Format the data according to the format required by IFEval's official evaluation script, then call their official evaluation script
    """
    results = []
    for example in responses:
        assert example['input'][-1]['role'] == 'user'
        instruction = example['input'][-1]['content']  # the last turn is the instruction
        prediction = extract_output(args, example)
        results.append({"prompt": instruction, "response": prediction})

    response_save_path = os.path.join(args.eval_output_dir, "ifeval_responses.jsonl")
    save_list_as_jsonl(response_save_path, results)

    evaluation_command = (
        f"python src/ifeval/evaluate/evaluation_main.py"
        f" --input_data={args.input}"
        f" --input_response_data={response_save_path}"
        f" --output_dir={args.eval_output_dir}"
    )

    print("*" * 30)
    print(evaluation_command)
    os.system(evaluation_command)
    