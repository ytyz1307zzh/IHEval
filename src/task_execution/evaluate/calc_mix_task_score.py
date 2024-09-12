"""
Calculate the score of each individual task from a mixed result file.
"""

import json
import argparse
from statistics import mean


parser = argparse.ArgumentParser()
parser.add_argument("-input", type=str, required=True, help='Path to the input data file')
args = parser.parse_args()

data = json.load(open(args.input, 'r', encoding='utf8'))
print(f"Loaded {len(data)} examples from {args.input}")

task2score = {}

for example in data:
    task = example['answer']['task']
    if task not in task2score:
        task2score[task] = {"strict": [], "loose": []}

    task2score[task]['strict'].append(example['strict_score'])
    task2score[task]['loose'].append(example['loose_score'])

for task, scores in task2score.items():
    strict_score = mean(scores['strict'])
    loose_score = mean(scores['loose'])
    print(f"Task {task}: strict score {strict_score:.2%}, loose score {loose_score:.2%}")
