"""
Evaluation function of Verb Extraction
"""
from collections import Counter
from string import punctuation
from copy import deepcopy


def eval_verb_extract(answer, prediction, loose=False):
    """
    For loose evaluation, remove the preceding explanation sentences
    """

    answer = answer.lower()
    prediction = prediction.lower()

    if loose:
        p = prediction.split("\n")
        prediction_remove_first = "\n".join(p[1:]).strip()
        prediction_remove_last = "\n".join(p[:-1]).strip()
        prediction_remove_both = "\n".join(p[1:-1]).strip()
        revised_prediction = prediction.replace("*", "")
        revised_prediction_remove_first = prediction_remove_first.replace("*", "")
        revised_prediction_remove_last = prediction_remove_last.replace("*", "")
        revised_prediction_remove_both = prediction_remove_both.replace("*", "")
        all_predictions = [
            prediction,
            revised_prediction,
            prediction_remove_first,
            prediction_remove_last,
            prediction_remove_both,
            revised_prediction_remove_first,
            revised_prediction_remove_last,
            revised_prediction_remove_both,
        ]

        all_prediction_scores = []
        for p in all_predictions:
            # Claude may generate an extra "verbs:" prefix
            if ":" in p:
                p = p.split(":")[1]
            all_prediction_scores.append(word_f1_no_punc(answer, p))
        return max(all_prediction_scores)

    else:
        return word_f1_no_punc(answer, prediction)
        

def word_f1_no_punc(answer, prediction):
    
    # Step 2: Convert "," to ", " and then remove all punctuation
    answer = answer.replace(",", ", ")
    prediction = prediction.replace(",", ", ")
    # remove excessive whitespaces
    answer = ' '.join(answer.split())
    prediction = ' '.join(prediction.split())
    answer = ''.join([c for c in answer if c not in punctuation])
    prediction = ''.join([c for c in prediction if c not in punctuation])

    # Step 3: Segment words by spaces
    answer_words = answer.split()
    prediction_words = prediction.split()

    # Calculate true positives using a Counter
    answer_counter = Counter(answer_words)
    prediction_counter = Counter(prediction_words)

    # Calculate true positives, false positives, and false negatives
    true_positives = sum((answer_counter & prediction_counter).values())
    false_positives = sum((prediction_counter - answer_counter).values())
    false_negatives = sum((answer_counter - prediction_counter).values())

    # Step 4: Calculate precision, recall, and F1 score
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1

# answer = "translate, following, convey, leaving, partnered, agreed, finance, installed, based, designed, had, issued"
# prediction = "Verbs: leaving, partnered, agreed, finance, installed, designed, had, issued"
# print(eval_verb_extract(answer, prediction, loose=True))
