"""
Aggregate scores from different tasks
"""

import json
import argparse
from statistics import mean

import colorama
from termcolor import colored
colorama.init()


def aggregate_task_score(score_dict):
    all_setting_scores = []
    for setting, setting_score_dict in score_dict.items():
        setting_avg_score = setting_score_dict['average']
        all_setting_scores.append(setting_avg_score)

    return mean(all_setting_scores)


parser = argparse.ArgumentParser()
parser.add_argument("-record", type=str, required=True, help='Path to the input data file')
parser.add_argument("-output", type=str, required=True, default="Path to the output data file")
args = parser.parse_args()

data = json.load(open(args.record, 'r', encoding='utf8'))
print(f"Loaded {len(data)} examples from {args.record}")

results = {}
for domain in data.keys():

    # For llama-3, skip the tool-use category because officially it does not support tool callling
    if (args.record.endswith("llama3-8b.json") or args.record.endswith("llama3-70b.json")) and domain == 'tool-use':
        continue

    for task, task_score_dict in data[domain].items():
        task_ref_score = aggregate_task_score(task_score_dict['reference'])
        task_align_score = aggregate_task_score(task_score_dict['aligned'])
        task_conflict_score = aggregate_task_score(task_score_dict['conflict'])

        results[f"{domain}_{task}"] = {
            "reference": task_ref_score,
            "aligned": task_align_score,
            "conflict": task_conflict_score,
        }

results['overall'] = {
    "reference": mean([results[task]['reference'] for task in results.keys()]),
    "aligned": mean([results[task]['aligned'] for task in results.keys()]),
    "conflict": mean([results[task]['conflict'] for task in results.keys()]),
}

json.dump(results, open(args.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
print(f"Saved Overall Results to {args.output}")

print()
for task, task_scores in results.items():
    if task == 'overall':
        continue
    print(colored(f"Task: {task}", "red"))
    print(colored(f"Reference: {task_scores['reference']:.1%}", "green"), end=", ")
    print(colored(f"Aligned: {task_scores['aligned']:.1%}", "green"), end=", ")
    print(colored(f"Conflict: {task_scores['conflict']:.1%}", "green"))
    print()

print(colored("Overall", "red"))
print(colored(f"Agg. Reference: {results['overall']['reference']:.1%}", "green"), end=", ")
print(colored(f"Agg. Aligned: {results['overall']['aligned']:.1%}", "green"), end=", ")
print(colored(f"Agg. Conflict: {results['overall']['conflict']:.1%}", "green"))
print(colored(f"Diff. Aligned: {results['overall']['aligned'] - results['overall']['reference']:.1%}", "yellow"), end=", ")
print(colored(f"Diff. Conflict: {results['overall']['conflict'] - results['overall']['reference']:.1%}", "yellow"))

# Average of absolute difference
print(colored(f"Absolute Diff. Aligned: {mean([abs(results[task]['aligned'] - results[task]['reference']) for task in results.keys() if task != 'overall']):.1%}", "yellow"))
print(colored(f"Absolute Diff. Conflict: {mean([abs(results[task]['conflict'] - results[task]['reference']) for task in results.keys() if task != 'overall']):.1%}", "yellow"))
