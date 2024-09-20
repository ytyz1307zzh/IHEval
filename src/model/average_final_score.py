"""
Aggregate scores from different tasks
"""

import pdb
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
    all_task_scores = {"reference": [], "aligned": [], "conflict": []}
    for task, task_score_dict in data[domain].items():
        task_ref_score = aggregate_task_score(task_score_dict['reference'])
        task_align_score = aggregate_task_score(task_score_dict['aligned'])
        task_conflict_score = aggregate_task_score(task_score_dict['conflict'])

        all_task_scores['reference'].append(task_ref_score)
        all_task_scores['aligned'].append(task_align_score)
        all_task_scores['conflict'].append(task_conflict_score)

    results[domain] = {
        "reference": mean(all_task_scores['reference']),
        "aligned": mean(all_task_scores['aligned']),
        "conflict": mean(all_task_scores['conflict']),
    }

results['overall'] = {
    "reference": mean([results[domain]['reference'] for domain in data.keys()]),
    "aligned": mean([results[domain]['aligned'] for domain in data.keys()]),
    "conflict": mean([results[domain]['conflict'] for domain in data.keys()]),
}

json.dump(results, open(args.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
print(f"Saved Overall Results to {args.output}")

print()
print(colored(f"Agg. Reference Score: {results['overall']['reference']}", "green"))
print(colored(f"Agg. Aligned Score: {results['overall']['aligned']}", "green"))
print(colored(f"Agg. Conflict Score: {results['overall']['conflict']}", "green"))
