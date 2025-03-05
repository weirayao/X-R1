from src.x_r1.reward_score.utils import extract_answer, normalize_text, evaluate_answer_similarity
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

def accuracy_answer_reward(completion, solution, **kwargs):
    """Reward function that checks if the completion matches the ground truth solution.
    
    Args:
        completion: A single completion dictionary containing the response
        solution: A single ground truth solution string
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Reward score between 0 and 1
    """
    # is it a completion dictionary in the format of {"content": "...", "role": "assistant"} or a string?
    assert isinstance(completion, str), "Completion must be a string"
    content = completion
    
    # First try latex parsing
    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        reward = float(verify(answer_parsed, gold_parsed))
        print('-'*100)
        print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
    else:
        # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
        answer_content = extract_answer(content)
        normalized_content = normalize_text(answer_content)
        normalized_solution = normalize_text(solution)
        reward = evaluate_answer_similarity(normalized_content, normalized_solution)
        print('-'*100)
        print('\nanswer_parsed:', normalized_content, '\ngold_parsed:', normalized_solution, '\nreward:', reward)

    print('\naccuracy reward:', reward)
    return reward