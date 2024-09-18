"""
General script of running open-source models on NLP tasks with instruction hierarchy
"""

import os
import pdb
import sys
import json
import random
import inspect

random.seed(42)
import argparse

sys.path.append(".")
import src.task_execution.evaluate as task_eval
import src.safety.evaluate as safety_eval
import src.rule_following.evaluate as rule_eval
import src.tool_use.evaluate as tool_eval
from tqdm import tqdm

eval_func_map = {
    "verb-extract": task_eval.eval_verb_extract,
    "translation": task_eval.eval_translation,
    "lang-detect": task_eval.eval_lang_detect,
    "secret-password": safety_eval.eval_tensortrust,
    "single-turn": rule_eval.eval_ifeval,
    "multi-turn": rule_eval.eval_ifeval,
    "get-webpage": task_eval.eval_mixed,
    "slack-user": tool_eval.eval_slack_user,
}

eval_metric_map = {
    "verb-extract": "F1",
    "translation": "ROUGE-L",
    "lang-detect": "Accuracy",
    "secret-password": "Accuracy",
    "get-webpage": "Overall",
    "slack-user": "Accuracy",
}

EXIT_EVAL_TASKS = ["single-turn", "multi-turn"]


def extract_output(args, example):
    if args.backend == "api":
        if "claude" in args.model:
            try:
                return example["output"]["content"][0]["text"].strip()
            except (IndexError, KeyError):
                return ""  # If Claude does not return any output, use empty string
        elif "gpt" in args.model:
            if example['output'] is not None:
                return example["output"].strip()
            else:
                return ""
        elif "llama" in args.model:
            return example["output"]['generation'].strip()
    else:
        return example["output"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", type=str, required=True, help='Huggingface model name or local path'
    )
    parser.add_argument(
        "-input",
        type=str,
        default="nlp-task/verb-extract/conflict/input_data_system_verb_extract_user_translate_qa.json",
    )
    parser.add_argument(
        "-request_file",
        type=str,
        default="nlp-task/verb-extract/conflict/qwen/qwen2-7b/qwen2-7b-requests.json",
    )
    parser.add_argument(
        "-response_file",
        type=str,
        default="nlp-task/verb-extract/conflict/qwen/qwen2-7b/qwen2-7b-responses.json",
    )
    parser.add_argument(
        "-eval_output_dir", type=str, default="nlp-task/verb-extract/conflict/qwen/"
    )
    parser.add_argument("-no_override_output", action="store_true", help='If set, do not override the output file')
    parser.add_argument("-max_tokens", type=int, default=1024)
    parser.add_argument("-top_k", type=int, default=250)
    parser.add_argument("-top_p", type=float, default=1.0)
    parser.add_argument("-temperature", type=float, default=0.0)
    parser.add_argument("-precision", type=str, default="auto")
    parser.add_argument(
        "-backend", type=str, choices=["vllm", "hf", "lmdeploy", "api"], required=True
    )
    parser.add_argument("-task", type=str, choices=eval_func_map.keys(), required=True)

    # API call arguments
    parser.add_argument("-max_loop", type=int, default=2)
    parser.add_argument("-max_retries", type=int, default=10)
    parser.add_argument("-max_threads", type=int, default=8)
    parser.add_argument("-sleep", type=int, default=2)
    args = parser.parse_args()

    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")

    # prepare model inputs
    requests = []
    id2answer = {}
    for example in data:
        id_ = example["id"]
        messages = []

        # For multi-turn settings, there should be a "conversation_history" field in the example
        if "conversation_history" in example:
            messages.extend([
                {"role": "user", "content": msg} if i % 2 == 0 else {"role": "assistant", "content": msg}
                for i, msg in enumerate(example["conversation_history"])
            ])

        messages.append({"role": "user", "content": example["instruction"]})

        request_example = {"id": id_}

        if "system" in example:
            if args.backend == "api":
                request_example["system"] = example["system"]
            else:
                messages.insert(0, {"role": "system", "content": example["system"]})

        request_example["messages"] = messages

        if "tool" in example:
            request_example["tool"] = example["tool"]

        requests.append(request_example)
        id2answer[id_] = example["answer"]

    # create directory if not exists
    os.makedirs(os.path.dirname(args.request_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.response_file), exist_ok=True)
    json.dump(
        requests,
        open(args.request_file, "w", encoding="utf-8"),
        indent=4,
        ensure_ascii=False,
    )

    # run inference command
    if args.backend == "vllm":
        inference_script = "src/utils/run_vllm_model.py"
    elif args.backend == "hf":
        inference_script = "src/utils/run_hf_model.py"
    elif args.backend == "lmdeploy":
        inference_script = "src/utils/run_lmdeploy_model.py"
    # "api": claude or gpt models
    elif args.backend == "api":
        inference_script = "src/utils/call_api.py"
    else:
        raise NotImplementedError

    inference_command = (
        f"python {inference_script}"
        f" -model {args.model}"
        f" -input {args.request_file}"
        f" -output {args.response_file}"
        f" -max_new_tokens {args.max_tokens}"
        f" -top_k {args.top_k}"
        f" -top_p {args.top_p}"
        f" -temperature {args.temperature}"
    )

    if args.backend == "api":
        inference_command += (
            f" -max_loop {args.max_loop}"
            f" -max_retries {args.max_retries}"
            f" -max_threads {args.max_threads}"
            f" -sleep {args.sleep}"
        )

    # Run inference (if inference was done before, the script will skip running the model)
    print("*" * 30)
    print(inference_command)
    os.system(inference_command)

    responses = json.load(open(args.response_file, "r", encoding="utf-8"))
    eval_func = eval_func_map[args.task]
    accept_loose_score = 'loose' in inspect.signature(eval_func).parameters

    if args.task in EXIT_EVAL_TASKS:
        eval_func(args, id2answer, responses, extract_output)
        return

    results = []
    strict_scores, loose_scores = [], []
    for example in tqdm(responses, desc="Evaluating"):
        id_ = example["id"]
        answer = id2answer[id_]
        prediction = extract_output(args, example)

        strict_score = round(eval_func(answer, prediction), 2)
        strict_scores.append(strict_score)

        save_input = example["input"]
        if "system" in example:
            save_input = [{"role": "system", "content": example["system"]}] + save_input

        result_dict = {
            "id": id_,
            "input": save_input,
            "answer": answer,
            "output": prediction,
            "strict_score": strict_score,
        }

        if accept_loose_score:
            loose_score = round(eval_func(answer, prediction, loose=True), 2)
            loose_scores.append(loose_score)
            result_dict["loose_score"] = loose_score

        results.append(result_dict)

    print(
        f"Strict {eval_metric_map[args.task]} score: {sum(strict_scores) / len(strict_scores):.2%}"
    )
    if accept_loose_score:
        print(
            f"Loose {eval_metric_map[args.task]} score: {sum(loose_scores) / len(loose_scores):.2%}"
        )

    # Save evaluation results if args.no_override_output is not set
    if not args.no_override_output:
        # If there are non-integer ids, sorting may fail
        try:
            results = sorted(results, key=lambda x: x["id"])
        except Exception as e:
            print(f'Skip sorting results due to error: {e}')

        os.makedirs(args.eval_output_dir, exist_ok=True)
        eval_output_path = os.path.join(args.eval_output_dir, f"eval_results.json")
        json.dump(
            results,
            open(eval_output_path, "w", encoding="utf-8"),
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
