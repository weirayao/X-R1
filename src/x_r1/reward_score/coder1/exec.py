import os
import subprocess
import signal
import resource
import tempfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

CLI_ARG_SIZE_LIMIT = 1024 * 3

def code_exec_direct(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    """
    Execute Python code directly using subprocess without firejail.
    Similar to the approach used in codeforce.py.
    """
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]  # avoid importing wrong stuff

    if pytest:
        # solution is in {tmpdir}/solution.py
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            # Write the solution to a file
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            
            command = ["python3", "-m", "pytest", tmpdir]
            try:
                result = subprocess.run(
                    command,
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired:
                return False, _ERROR_MSG_PREFIX + "Execution timed out"
    else:
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command = ["python3", "-c", code]
            try:
                result = subprocess.run(
                    command,
                    input=stdin.encode() if stdin else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired:
                return False, _ERROR_MSG_PREFIX + "Execution timed out"
        else:
            with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                tmp_path = tmp.name
            
            try:
                command = ["python3", tmp_path]
                result = subprocess.run(
                    command,
                    input=stdin.encode() if stdin else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                    timeout=timeout
                )
                # Clean up the temporary file
                os.unlink(tmp_path)
            except subprocess.TimeoutExpired:
                # Clean up the temporary file even if timeout occurs
                os.unlink(tmp_path)
                return False, _ERROR_MSG_PREFIX + "Execution timed out"
            except Exception as e:
                # Clean up the temporary file if any other exception occurs
                os.unlink(tmp_path)
                return False, _ERROR_MSG_PREFIX + f"Error: {str(e)}"

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"