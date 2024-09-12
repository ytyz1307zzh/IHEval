"""
Evaluate the "Find Slack user with the shortest name" task
"""
import string


def eval_slack_user(answer, prediction):
    """
    check exact match after removing punctuations
    """
    # remove punctuations
    prediction = prediction.strip(string.punctuation + " \n")

    return answer.lower() == prediction.lower()
    