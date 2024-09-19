"""
Record all scores of the same model into a single JSON file.
"""

import os
import json
import argparse


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

    return {
        "prompt_strict": strict_prompt_score,
        "instruction_strict": strict_instruction_score,
        "prompt_loose": loose_prompt_score,
        "instruction_loose": loose_instruction_score
    }


def record_regular_task(data):
    strict_scores, loose_scores = [], []

    for example in data:
        if "strict_score" in example:
            strict_scores.append(example["strict_score"])
        if "loose_score" in example:
            loose_scores.append(example["loose_score"])

    score_dict = {}
    if len(strict_scores) > 0:
        score_dict['strict_score'] = sum(strict_scores) / len(strict_scores)
    if len(loose_scores) > 0:
        score_dict['loose_score'] = sum(loose_scores) / len(loose_scores)

    return score_dict
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, required=True, help='Path to eval_results.json')
    parser.add_argument('-domain', type=str, help='Domain of the task')
    parser.add_argument('-task', type=str, help='task name')
    parser.add_argument('-prompt_setting', type=str, help='prompt setting (strong/weak prompts)')
    parser.add_argument('-instruction_type', type=str, help='conflict/aligned/reference')
    parser.add_argument('-model', type=str, help='Model name')
    parser.add_argument('-output_dir', type=str, default='record/', help='Directory to save the output')
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, f'{args.model}.json')

    all_scores = []
    if os.path.exists(output_path):
        all_scores = json.load(open(output_path, 'r', encoding='utf-8'))
    
    if args.domain == 'rule-following':
        score_dict = record_rule_following(data)

    else:
        data = json.load(open(args.data, 'r', encoding='utf-8'))
        print(f'Loaded {len(data)} data instances from {args.data}')
        score_dict = record_regular_task(data)

    all_scores[args.domain][args.task][args.instruction_type][args.prompt_setting] = score_dict

    json.dump(all_scores, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print('*' * 30 + '\n' + f'Saved scores to {output_path}')


if __name__ == '__main__':
    main()

