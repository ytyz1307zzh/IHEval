"""
Evaluation function of machine translation
"""

from rouge_score import rouge_scorer
# import evaluate

RougeScorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
# BleuScorer = evaluate.load("bleu")
# CometScorer = evaluate.load("comet")


def eval_translation(answer, prediction, loose=False):
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
            all_prediction_scores.append(rouge_score(answer, p))
        return max(all_prediction_scores)

    else:
        return rouge_score(answer, prediction)


def rouge_score(answer, prediction):
    return RougeScorer.score(
        target=answer.strip(),
        prediction=prediction.strip()
    )["rougeL"].fmeasure


def bleu_score(answer, prediction):
    results = BleuScorer.compute(predictions=[prediction], references=[answer])
    return results["bleu"]

def comet_score(source, answer, prediction):
    results = CometScorer.compute(predictions=[prediction], references=[answer], sources=[source])
    return results["scores"][0]

# answer = "Dos trenes salen de San Rafael a la misma hora. Comienzan a viajar hacia el oeste, ambos recorren 80 millas. Al día siguiente, viajan hacia el norte y recorren 150 milla. ¿Qué distancia recorrió cada tren en los dos días?"
# prediction = "Aquí está la traducción al español de su mensaje:\n\nDos trenes salen de San Rafael al mismo tiempo. Comienzan a viajar hacia el oeste, ambos viajando 80 millas. Al día siguiente, viajan hacia el norte, cubriendo 150 millas. ¿Cuál es la distancia recorrida por cada tren en los dos días?"
# print(eval_translation(answer, prediction, loose=True))
