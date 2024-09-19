"""
Calculate the reference score for the translation task
Reference score: the normal translation performance (translating user instruction + math question)
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.append(".")
import src.task_execution.evaluate as task_eval
from statistics import mean

eval_func_map = {
    "verb-extract": task_eval.eval_verb_extract,
    "translation": task_eval.eval_translation,
}

eval_metric_map = {
    "verb-extract": "F1",
    "translation": "ROUGE-L",
}

instruction_data_separator_map = {
    "verb-extract": ", ",
    "translation": "\n",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input", type=str, default="reference/llama/llama3-8b/eval_results.json"
    )
    parser.add_argument("-task", type=str, choices=eval_func_map.keys(), required=True)
    parser.add_argument("-record_dir", type=str, default="model-scores/")
    args = parser.parse_args()

    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")

    strong_instruction_reference = None
    strong_instruction_prediction = None
    weak_instruction_reference = None
    weak_instruction_prediction = None

    # Find the reference and prediction for strong and weak user instructions first
    # Because the JSON list may not be sorted by ID
    for example in tqdm(data):
        id_ = example["id"]
        reference = example["answer"]
        prediction = example["output"]

        # Remove the prefix for the translation task
        if prediction.startswith("español:"):
            prediction = prediction[len("español:") :].strip()

        # Remove the prefix for the verb extraction task
        if prediction.startswith("Verbs:"):
            prediction = prediction[len("Verbs:") :].strip()

        if id_ == "strong_user_instruction":
            strong_instruction_reference = reference
            strong_instruction_prediction = prediction

        elif id_ == "weak_user_instruction":
            weak_instruction_reference = reference
            weak_instruction_prediction = prediction

    strong_strict_scores, strong_loose_scores = [], []
    weak_strict_scores, weak_loose_scores = [], []
    data_strict_scores, data_loose_scores = [], []  # no user instruction, only translate the math question itself

    # Construct the whole sequence (user instruction + math question) for each instance
    eval_func = eval_func_map[args.task]

    for example in data:
        id_ = example["id"]
        reference = example["answer"]
        prediction = example["output"]

        if id_ == "strong_user_instruction" or id_ == "weak_user_instruction":
            continue

        # Remove the prefix for the translation task
        if prediction.startswith("español:"):
            prediction = prediction[len("español:") :].strip()

        # Remove the prefix for the verb extraction task
        if prediction.startswith("Verbs:"):
            prediction = prediction[len("Verbs:") :].strip()

        separator = instruction_data_separator_map[args.task]

        whole_strong_reference = strong_instruction_reference + separator + reference
        whole_strong_prediction = strong_instruction_prediction + separator + prediction
        whole_weak_reference = weak_instruction_reference + separator + reference
        whole_weak_prediction = weak_instruction_prediction + separator + prediction

        strong_strict_score = round(
            eval_func(whole_strong_reference, whole_strong_prediction), 2
        )
        strong_loose_score = round(
            eval_func(whole_strong_reference, whole_strong_prediction, loose=True), 2
        )

        weak_strict_score = round(
            eval_func(whole_weak_reference, whole_weak_prediction), 2
        )
        weak_loose_score = round(
            eval_func(whole_weak_reference, whole_weak_prediction, loose=True), 2
        )

        data_strict_score = round(
            eval_func(reference, prediction), 2
        )
        data_loose_score = round(
            eval_func(reference, prediction, loose=True), 2
        )

        strong_strict_scores.append(strong_strict_score)
        strong_loose_scores.append(strong_loose_score)
        weak_strict_scores.append(weak_strict_score)
        weak_loose_scores.append(weak_loose_score)
        data_strict_scores.append(data_strict_score)
        data_loose_scores.append(data_loose_score)

    print(
        f"Strong user instruction strict {eval_metric_map[args.task]} score: {sum(strong_strict_scores) / len(strong_strict_scores):.2%}"
    )
    print(
        f"Strong user nstruction loose {eval_metric_map[args.task]} score: {sum(strong_loose_scores) / len(strong_loose_scores):.2%}"
    )
    print(
        f"Weak user instruction strict {eval_metric_map[args.task]} score: {sum(weak_strict_scores) / len(weak_strict_scores):.2%}"
    )
    print(
        f"Weak user instruction loose {eval_metric_map[args.task]} score: {sum(weak_loose_scores) / len(weak_loose_scores):.2%}"
    )
    print(
        f"No user instruction strict {eval_metric_map[args.task]} score: {sum(data_strict_scores) / len(data_strict_scores):.2%}"
    )
    print(
        f"No user instruction loose {eval_metric_map[args.task]} score: {sum(data_loose_scores) / len(data_loose_scores):.2%}"
    )

    # Record the model scores
    domain, task, instruction_type, prompt_setting, model_family, model, data_filename = args.input.split('/')[-7:]
    score_filepath = os.path.join(args.record_dir, f"{model}.json")

    all_scores = {}
    if os.path.exists(score_filepath):
        all_scores = json.load(open(score_filepath, 'r', encoding='utf-8'))

    score_dict = {
        "strong_user_instruction_strict": round(sum(strong_strict_scores) / len(strong_strict_scores), 4),
        "strong_user_instruction_loose": round(sum(strong_loose_scores) / len(strong_loose_scores), 4),
        "weak_user_instruction_strict": round(sum(weak_strict_scores) / len(weak_strict_scores), 4),
        "weak_user_instruction_loose": round(sum(weak_loose_scores) / len(weak_loose_scores), 4),
        "no_user_instruction_strict": round(sum(data_strict_scores) / len(data_strict_scores), 4),
        "no_user_instruction_loose": round(sum(data_loose_scores) / len(data_loose_scores), 4),
    }
    score_dict['average'] = round(mean(score_dict.values()), 4)

    if domain not in all_scores:
        all_scores[domain] = {}
    if task not in all_scores[domain]:
        all_scores[domain][task] = {}
    if instruction_type not in all_scores[domain][task]:
        all_scores[domain][task][instruction_type] = {}
        
    all_scores[domain][task][instruction_type][prompt_setting] = score_dict

    json.dump(all_scores, open(score_filepath, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'Saved scores to {score_filepath}')


if __name__ == "__main__":
    main()
