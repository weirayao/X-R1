import re
import unittest
import tempfile
import subprocess
import concurrent.futures
import textwrap
import random
import os

execution_block = textwrap.dedent("""
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCases)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    passed_tests = total_tests - failed_tests
    print(f"{passed_tests},{total_tests}")
""")

def run_test(test_script):
    """Runs a test script for a given function."""
    # test_script = test_script_template.format(model_function=model_function)
    # Save test script to a temporary file
    custom_temp_dir = "/export/home/shirley/temp_dir/"
    
    current_dir = os.getcwd()  # Store the original directory
    os.chdir(custom_temp_dir)  # Switch to temp dir
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=custom_temp_dir) as test_file:
        test_file.write(test_script)
        test_file_path = test_file.name

    # Execute test script
    result = subprocess.run(
            ["python3", "-c", test_script],  # Run code as a separate Python process
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Ensure output is returned as a string
            timeout=10  # Set a timeout (in seconds)
        )
    
    os.chdir(current_dir)
    
    # Return formatted results
    return result.stdout


def extract_solution(model_solution, unit_test):
    try:
        
        result = run_test(unit_test + "\n" + model_solution +"\n" + execution_block)
            
        result = result.splitlines()[-1]
        passed_tests, total_tests = result.replace("\n", "").split(",")
        passed_tests = int(passed_tests)
        total_tests = int(total_tests)
        
        return passed_tests/total_tests
    except:
        return None
    
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
    score=1.
    unit_test = ground_truth["unit_test"]
    model_solution = ground_truth["answer"]
    code_prompt = ground_truth["code_prompt"]
    
    solution_str = process_model_solution(solution_str)
    
    do_print = random.randint(1, 5) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"")
        
    if not solution_str:
        if do_print:
            print(f"Python Markdown not found")
        return format_score
    
    try:
        result = extract_solution(solution_str, unit_test)
        if result is None:
            if do_print:
                print(f"Could not evaluate code")
            return -0.5
            
        if int(result) == 1:  # Account for floating point precision
            if do_print:
                print(f"ALL UNIT-TESTS PASSED")
            return score
        else:
            if do_print:
                print(f"PARTIAL UNIT-TESTS PASSED")
            return result
    except:
        if do_print:
            print(f"Error evaluating code")
        return format_score 