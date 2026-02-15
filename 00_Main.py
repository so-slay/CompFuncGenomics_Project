import subprocess
import sys 

# List the scripts in the order you want them to run
scripts = [
    ["GetFASTAfromTSV.py", sys.argv[1]],
    "markovNull.py",
    "CrossValidationScores.py"  
    "Plots.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    # Print the script's output
    print(result.stdout)
    
    # Print errors if any
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
    
    # Optional: stop if a script fails
    if result.returncode != 0:
        print(f"{script} failed with exit code {result.returncode}. Stopping execution.")
        break
