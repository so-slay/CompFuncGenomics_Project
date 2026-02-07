import sys 
import os 
from pathlib import Path

import pandas as pd

# Section 1: read CONFIG file

# Function to parse details from config file
def read_config(path):
    config = {}

    with open(path, "r") as f:
        for line in f: 
            line = line.strip()
        # Ignore empty lines/comments (start with #)
            if not line or line.startswith("#"):
                continue
            key,val = line.split("=", 1)   
            config[key.strip()] = val.strip()
    return config

# Pull information from specified config file 
config_path = sys.argv[1]
config = read_config(config_path)

input_dir = Path(config["input_dir"])
if not input_dir.exists() or not input_dir.is_dir(): 
    print("FATAL: No input directory specified")
    sys.exit(1)

files = config.get("files")

if files:
    files = [f.strip() for f in files.split(",")]
    tsv = []
    for f in files: 
        if f.endswith(".tsv"):
            tsv.append(f)
        else:
            print(f"Ignoring '{f}', not a .tsv file")
    files = tsv
else: 
    files = [f.name for f in input_dir.glob("*.tsv")]


if not files: 
    print("FATAL: No tsv files found in specified input_dir")
    sys.exit(1)

# Section 2: Convert tsv to BED file
for fname in files:
    input_tsv = input_dir / fname
    output_tsv = os.path.splitext(input_tsv)[0] + '.bed'
    df = pd.read_csv(input_tsv, sep="\t")

    # Remove header, save in BED-like tab sep format.
    # write output as headerless bed file
    with open(f"{output_tsv}", "w") as f:
        df.to_csv(f, sep="\t", header=False, index=False)



# Extra Credit:

# Write the headers of the files to a log file