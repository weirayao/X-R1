import re

def compute_score(solution_str, ground_truth, method='strict', flex_score=0.1, match_answer_score=0.9, score=1.0):
    """The scoring function for LsatQA.
    # TODO: add reference score such that the score is greater than 0 if the solution is from one of the references. 
    
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        score: the score for the correct answer
    """
    solution_str = extract_solution(solution_str)
    if is_equiv(solution_str, ground_truth):
        return score
    elif match_answer_with_format_strip(solution_str, ground_truth):
        return match_answer_score
    elif contain_answer(solution_str, ground_truth):
        if method == 'strict':
            return 0.0
        else:
            return flex_score
    else:
        return 0.0

def extract_solution(text):
    """Extract the answer from the solution string."""
    # # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    match = re.search(r'####\s*(.+)', text)
    if match:
        solution_str = match.group(1)
    else:
        # If no match, return the original string
        return solution_str

def strip_string(string):
    string = string.strip()
    return string

def format_strip(string):
    # string <> signs
    string = string.strip('<>').strip()
    # string [] signs
    string = string.strip('[]').strip()
    # string () signs
    string = string.strip('()').strip()
    # string {} signs
    string = string.strip('{}').strip()
    # string '' signs
    string = string.strip("'").strip()
    # string "" signs
    string = string.strip('"').strip()
    return string


def is_equiv(str1, str2, verbose=False):
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
    
def contain_answer(str1, str2, verbose=False):
    # ground truth is strings and solution is a string that contains the answer
    # we need to check if the solution contains the ground truth
    str1 = strip_string(str1)
    str2 = strip_string(str2)
    if verbose:
        print(str1, str2)
    if str2 in str1:
        return True
    else:
        return False

def match_answer_with_format_strip(answer_str, ground_truth, verbose=False):
    try:
        answer_str = format_strip(answer_str)
        ground_truth = format_strip(ground_truth)
        return answer_str==ground_truth
    except Exception:
        return answer_str==ground_truth
