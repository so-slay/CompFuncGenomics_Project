import subprocess
import sys 
import time

start = time.time()
config_path = sys.argv[1] if len(sys.argv) > 1 else "01Default_config.txt"
# List the scripts in the order you want them to run
scripts = [
    ["GetFASTAfromTSV.py", config_path],
    ["Plots.py"]
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", *script], capture_output=True, text=True)

    # Print the script's output
    print(result.stdout)
    
    # Print errors if any
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
    
    # Optional: stop if a script fails
    if result.returncode != 0:
        print(f"{script} failed with exit code {result.returncode}. Stopping execution.")
        break

end  = time.time()


print(f"Exiting script. Runtime: {end-start} seconds.")
print(f"What's the difference between an alligator and a crocodile? One sees you later, the other, in a while")


