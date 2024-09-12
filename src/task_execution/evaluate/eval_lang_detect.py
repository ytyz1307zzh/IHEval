"""
Evaluation function of language detection
"""
import re
import json

def eval_lang_detect(answer, prediction):
    """
    "answer" should be the name of a language, or a list of acceptable language names
    """
    if isinstance(answer, str):
        return language_accuracy(answer, prediction)
    elif isinstance(answer, list):
        return any([language_accuracy(ans, prediction) for ans in answer])
    else:
        raise TypeError(f"answer should be a string or a list of strings, but got {answer}")


def language_accuracy(answer, prediction):
    """
    Find the answer contained in a JSON obejct in the answer string.
    """

    if prediction.count("{") == 1 and prediction.count("}") == 0:
        prediction = prediction + "}"

    # Search for the JSON object in the prediction
    prediction = prediction.replace("\n", " ")  # remove linebreaks for better regex matching
    json_pattern = re.compile(r'\{.+?\}')
    match = json_pattern.findall(prediction)

    # If there is no or more than one JSON object in the prediction, directly consider as failed
    if len(match) != 1:
        return False
    
    json_str = match[0]
    try:
        # Parse the JSON string to a dictionary
        json_obj = json.loads(json_str)
        lang_pred = json_obj.get("language", "")

        # Check if other keys besides "language" exist in the JSON object
        if len(json_obj) > 1:
            return False

        if lang_pred.lower().strip() == answer.lower():
            return True
        else:
            return False
        
    except json.JSONDecodeError:
        # Found JSON-like text but failed to decode it
        return False


# def language_accuracy(answer, prediction):
#     # Check "english", "chinese", "spanish", "español" and "中文" only exists at most 1 in prediction
#     option_exist = [opt in prediction for opt in ["english", "chinese", "spanish", "español", "中文"]]
#     if sum(option_exist) > 1:
#         return False
    
#     if answer == "english":
#         return answer in prediction
#     elif answer == "chinese":
#         return "中文" in prediction or answer in prediction
#     elif answer == "spanish":
#         return "español" in prediction or answer in prediction
