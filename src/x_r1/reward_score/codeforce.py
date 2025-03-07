import sys
import io 
import re
import random
import os 
import tempfile
import signal
import textwrap
import subprocess

def score_model_outputs(unit_test_outputs, model_outputs):
    if len(model_outputs) != len(unit_test_outputs):
        return 0
    
    passed = 0
    for uo,mo in zip(unit_test_outputs, model_outputs):
        if mo:
            if str(uo) == str(mo):
                passed += 1
    
    return passed/len(unit_test_outputs)

# Define a handler for the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out!")
            

def check_rewards(inputs, model_solution):
    output = []
    custom_temp_dir = "/home/skokane/tmp/"

    current_dir = os.getcwd()  # Store the original directory
    os.chdir(custom_temp_dir)
    # Code from run_code
    run_code = textwrap.dedent(f"""
    for sample_value in {inputs}:
    """)
    code_to_execute = model_solution.replace("input()", "sample_value")
    
    # Add indents to the `code` by prepending 4 spaces to each line
    indented_code = textwrap.indent(code_to_execute, '    ')
    
    # Combine the `run_code` and the indented `code`
    full_code = run_code + indented_code
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=custom_temp_dir) as test_file:
        test_file.write(full_code)
        test_file_path = test_file.name
    
    # Execute test script
    result = subprocess.run(
            ["python3", "-c", full_code],  # Run code as a separate Python process
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Ensure output is returned as a string
            timeout=10  # Set a timeout (in seconds)
        )
    
        # Collect the captured output
    output = result.stdout.strip("\n").split("\n")
    
    os.chdir(current_dir)
    
    return output

def process_model_solution(solution_str):
    solution_str = solution_str.replace("\t", "    ")
    code = None
    if "```python" in solution_str:
        code = re.findall(r"```python\n(.*?)\n```", solution_str, re.DOTALL)
    elif "```Python" in solution_str:
        code = re.findall(r"```Python\n(.*?)\n```", solution_str, re.DOTALL)
    else:
        code = re.findall(r"```(.*?)```", solution_str, re.DOTALL)
    if code:
        if len(code[-1]) > 0:
            return code[-1]
        else:
            return code[0]
    return solution_str

def compute_score(solution_str, ground_truth):
    method='strict'
    format_score=-1
    score=1
    
    unit_test_inputs = list(ground_truth["unit_test"]["inputs"])
    unit_test_outputs = list(ground_truth["unit_test"]["outputs"])
    model_solution = ground_truth["answer"]
    
    solution_str = process_model_solution(solution_str)
    
    do_print = random.randint(1, 2) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"")
        
    if not solution_str:
        if do_print:
            print(f"Python Markdown not found")
        return format_score
    
    try:
        model_outputs = check_rewards(unit_test_inputs, solution_str)
        result = score_model_outputs(unit_test_outputs, model_outputs)
        if result is None:
            if do_print:
                print(f"Could not evaluate code")
            return -0.5
            
        if int(result) == 1:  # Account for floating point precision
            if do_print:
                print(f"CODEFORCE ALL UNIT-TESTS PASSED")
            return score
        else:
            if do_print:
                print(f"CODEFORCE PARTIAL UNIT-TESTS PASSED")
            return result
    except:
        if do_print:
            print(f"Error evaluating codeforce")
        return format_score 
    