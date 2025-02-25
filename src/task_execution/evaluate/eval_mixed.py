from . import eval_lang_detect, eval_translation, eval_verb_extract
import inspect

eval_func_map = {
    "verb_extract": eval_verb_extract,
    "translation": eval_translation,
    "lang_detect": eval_lang_detect,
}


def eval_mixed(answer, prediction, loose=False):
    task = answer['task']

    eval_func = eval_func_map[task]
    accept_loose_score = 'loose' in inspect.signature(eval_func).parameters
    if loose and accept_loose_score:
        return eval_func(answer['content'], prediction, loose=True)
    else:
        return eval_func(answer['content'], prediction)
