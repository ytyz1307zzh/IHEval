"""
Calculate the reference score for a mixed list of task examples.
"""

import os
import re
import pdb
import json
import argparse
from statistics import mean


parser = argparse.ArgumentParser()
parser.add_argument("-input", type=str, required=True, help='Path to the input data file')
parser.add_argument("-record_dir", type=str, default="model-scores/")
args = parser.parse_args()
assert args.input.endswith("eval_results.json")

data = json.load(open(args.input, 'r', encoding='utf8'))
print(f"Loaded {len(data)} examples from {args.input}")

verb_extraction_results, translation_results, language_results = [], [], []
for example in data:
    id_ = example['id']
    example['answer'] = example['answer']['content']

    if id_ == "verb_extraction_strong_tool_instruction":
        example['id'] = "strong_user_instruction"
        verb_extraction_results.append(example)

    elif id_ == "verb_extraction_weak_tool_instruction":
        example['id'] = "weak_user_instruction"
        verb_extraction_results.append(example)

    elif id_ == "translation_strong_tool_instruction":
        example['id'] = "strong_user_instruction"
        translation_results.append(example)

    elif id_ == "translation_weak_tool_instruction":
        example['id'] = "weak_user_instruction"
        translation_results.append(example)

    elif id_.startswith("verb_extraction"):
        example['id'] = int(re.match(r"verb_extraction_(.*)", id_).group(1))
        verb_extraction_results.append(example)

    elif id_.startswith("translation"):
        example['id'] = int(re.match(r"translation_(.*)", id_).group(1))
        translation_results.append(example)

    elif id_.startswith("language"):
        example['id'] = int(re.match(r"language_(.*)", id_).group(1))
        language_results.append(example)

print(f"Verb extraction examples: {len(verb_extraction_results)}, Translation examples: {len(translation_results)}, Language detection examples: {len(language_results)}")
verb_output_path = args.input.replace("eval_results.json", "eval_results_verb_extract.json")
translation_output_path = args.input.replace("eval_results.json", "eval_results_translation.json")
json.dump(verb_extraction_results, open(verb_output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
json.dump(translation_results, open(translation_output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
print(f"Saved verb extraction examples to {verb_output_path}")
print(f"Saved translation examples to {translation_output_path}")

os.makedirs(os.path.join(args.record_dir, 'verb-extract'), exist_ok=True)
verb_extract_command = (
    f"python src/task_execution/evaluate/calc_reference_score.py"
    f" -input {verb_output_path}"
    f" -task verb-extract"
    f" -record_dir {os.path.join(args.record_dir, 'verb-extract')}"
)

os.makedirs(os.path.join(args.record_dir, 'translation'), exist_ok=True)
translation_command = (
    f"python src/task_execution/evaluate/calc_reference_score.py"
    f" -input {translation_output_path}"
    f" -task translation"
    f" -record_dir {os.path.join(args.record_dir, 'translation')}"
)

print('*' * 30 + " Verb extraction " + '*' * 30)
os.system(verb_extract_command)
print('*' * 30 + " Translation " + '*' * 30)
os.system(translation_command)

print('*' * 30 + " Language detection " + '*' * 30)
language_accurary_list = [x['strict_score'] for x in language_results]
print(f"Data strict Accuracy score: {mean(language_accurary_list):.2%}")

# Record the scores
domain, task, instruction_type, prompt_setting, model_family, model, data_filename = args.input.split('/')[-7:]
score_filepath = os.path.join(args.record_dir, f"{model}.json")

all_scores = {}
if os.path.exists(score_filepath):
    all_scores = json.load(open(score_filepath, 'r', encoding='utf-8'))

verb_extract_score_dict = json.load(open(os.path.join(args.record_dir, 'verb-extract', f"{model}.json"), 'r', encoding='utf-8'))
translation_score_dict = json.load(open(os.path.join(args.record_dir, 'translation', f"{model}.json"), 'r', encoding='utf-8'))
verb_extract_avg_ref_score = verb_extract_score_dict[domain][task][instruction_type][prompt_setting]['average']
translation_avg_ref_score = translation_score_dict[domain][task][instruction_type][prompt_setting]['average']
# -2 is because there are instructions in the results besides the data
total_avg_score = ((len(verb_extraction_results) - 2) * verb_extract_avg_ref_score + (len(translation_results) - 2) * translation_avg_ref_score + len(language_results) * mean(language_accurary_list)) / (len(verb_extraction_results) + len(translation_results) - 2 + len(language_results) - 2)

score_dict = {
    "verb_extract_average": verb_extract_avg_ref_score,
    "translation_average": translation_avg_ref_score,
    "lang_detect_average": mean(language_accurary_list),
    "average": round(total_avg_score, 4)
}

print('*' * 30 + " Language detection " + '*' * 30)
print(f'Verb extraction average reference score: {verb_extract_avg_ref_score:.2%}')
print(f'Translation average reference score: {translation_avg_ref_score:.2%}')
print(f'Language detection average reference score: {mean(language_accurary_list):.2%}')
print(f'Overall average reference score: {total_avg_score:.2%}')

if domain not in all_scores:
    all_scores[domain] = {}
if task not in all_scores[domain]:
    all_scores[domain][task] = {}
if instruction_type not in all_scores[domain][task]:
    all_scores[domain][task][instruction_type] = {}
    
all_scores[domain][task][instruction_type][prompt_setting] = score_dict

# create a backup
json.dump(all_scores, open(score_filepath, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print(f'Saved scores to {score_filepath}')

# Remove intermediate files
os.remove(os.path.join(args.record_dir, 'verb-extract', f"{model}.json"))
os.remove(os.path.join(args.record_dir, 'translation', f"{model}.json"))
os.rmdir(os.path.join(args.record_dir, 'verb-extract'))
os.rmdir(os.path.join(args.record_dir, 'translation'))
