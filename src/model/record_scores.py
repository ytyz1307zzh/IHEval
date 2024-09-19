"""
Record all scores of the same model into a single JSON file.
"""

import os
import json
import argparse
from statistics import mean


def record_rule_following_single_file(data):

    prompt_total, instruction_total = 0, 0
    prompt_correct, instruction_correct = 0, 0

    for example in data:
        follow_instruction_list = example["follow_instruction_list"]
        instruction_id_list = example["instruction_id_list"]

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

    prompt_accuracy = prompt_correct / prompt_total
    instruction_accuracy = instruction_correct / instruction_total

    return prompt_accuracy, instruction_accuracy


def record_rule_following(filepath):
    assert filepath.endswith('eval_results.json')
    strict_filepath = filepath.replace('eval_results.json', 'eval_results_strict.json')
    loose_filepath = filepath.replace('eval_results.json', 'eval_results_loose.json')

    strict_data = json.load(open(strict_filepath, 'r', encoding='utf-8'))
    print(f'Loaded {len(strict_data)} data instances from {strict_filepath}')
    strict_prompt_score, strict_instruction_score = record_rule_following_single_file(strict_data)

    loose_data = json.load(open(loose_filepath, 'r', encoding='utf-8'))
    print(f'Loaded {len(loose_data)} data instances from {loose_filepath}')
    loose_prompt_score, loose_instruction_score = record_rule_following_single_file(loose_data)

    score_dict = {
        "prompt_strict": round(strict_prompt_score, 4), 
        "instruction_strict": round(strict_instruction_score, 4), 
        "prompt_loose": round(loose_prompt_score, 4), 
        "instruction_loose": round(loose_instruction_score, 4), 
    }
    score_dict['average'] = round(mean(score_dict.values()), 4)
    return score_dict


def record_regular_task(data):
    strict_scores, loose_scores = [], []

    for example in data:
        if "strict_score" in example:
            strict_scores.append(example["strict_score"])
        if "loose_score" in example:
            loose_scores.append(example["loose_score"])

    score_dict = {}
    if len(strict_scores) > 0:
        score_dict['strict_score'] = round(sum(strict_scores) / len(strict_scores), 4)
    if len(loose_scores) > 0:
        score_dict['loose_score'] = round(sum(loose_scores) / len(loose_scores), 4)

    score_dict['average'] = round(mean(score_dict.values()), 4)

    return score_dict
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, required=True, help='Path to eval_results.json')
    parser.add_argument('-output_dir', type=str, default='record/', help='Directory to save the output')
    args = parser.parse_args()

    domain, task, instruction_type, prompt_setting, model_family, model, data_filename = args.data.split('/')[-7:]
    output_path = os.path.join(args.output_dir, f'{model}.json')

    all_scores = {}
    if os.path.exists(output_path):
        all_scores = json.load(open(output_path, 'r', encoding='utf-8'))
    
    if domain == 'rule-following':
        score_dict = record_rule_following(args.data)

    else:
        data = json.load(open(args.data, 'r', encoding='utf-8'))
        print(f'Loaded {len(data)} data instances from {args.data}')
        score_dict = record_regular_task(data)

    if domain not in all_scores:
        all_scores[domain] = {}
    if task not in all_scores[domain]:
        all_scores[domain][task] = {}
    if instruction_type not in all_scores[domain][task]:
        all_scores[domain][task][instruction_type] = {}

    all_scores[domain][task][instruction_type][prompt_setting] = score_dict

    json.dump(all_scores, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'Saved scores to {output_path}')


if __name__ == '__main__':
    main()

