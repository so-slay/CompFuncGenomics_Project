import subprocess
import sys 
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
config_path = sys.argv[1] if len(sys.argv) > 1 else "01Default_config.txt"
config = read_config(config_path)
print(config)
ref_genome = Path(config.get("ref_genome"))
input_dir = Path(config["input_dir"] or ".").resolve()
if not input_dir.is_dir(): 
    print(f"FATAL: specified input is not a dir: {input_dir}")
    sys.exit(1)

files = config.get("files")
# default is to get all .tsv in specified dir
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

# Check for existing BED files: 

# Check existence of .bed and .fasta files for each TSV
files_to_convert = []
files_to_fetch = []

for fname in files:
    bed_file = input_dir / Path(fname).with_suffix(".bed")
    fasta_file = Path("FASTAs") / Path(fname).with_suffix(".fa")
    
    if bed_file.exists() and fasta_file.exists():
        continue
    elif fasta_file.exists() and not bed_file.exists():
        files_to_convert.append(fname)
    else:
        print(f"Fetching FASTAs from reference genome at {config.get("ref_genome")} for {fname}")
        files_to_convert.append(fname)  # Need BED first
        files_to_fetch.append(fname)

# Section 2: Convert tsv to BED file
for fname in files_to_convert:
    input_tsv = input_dir / fname
    output_tsv = input_tsv.with_suffix(".bed") 
    df = pd.read_csv(input_tsv, sep="\t")

    # Remove header, save in BED-like tab sep format.
    # write output as headerless bed file
    with open(f"{output_tsv}", "w") as f:
        df.to_csv(f, sep="\t", header=False, index=False)
    

# Extra Credit:
# Write the headers of the files to a log file

# Section 3: Fetch Sequence from Reference Genome using bedtools getfasta
# Write output into separate dir.

bash_script = Path(__file__).parent / "GetFastaFromBED.sh"

bed_files = [Path(f).with_suffix(".bed").name for f in files_to_fetch]


def bedtools_getfasta(input_files):
    output_dir = Path("FASTAs")
    output_dir.mkdir(exist_ok=True)

    for i in input_files:
        input_bed = input_dir / i
        output_fa = output_dir / f'{input_bed.stem}.fa'
        print("input_bed:", input_bed)
        print("exists:", input_bed.exists())

        
        try:
            subprocess.run([bash_script, input_bed, output_fa, ref_genome], check=1)
        except subprocess.CalledProcessError as e:
            print(f"error in file {i}: {e}")
    print(f"FASTAs written to {output_dir} if not already present")
        

if __name__ == "__main__":
    
    bedtools_getfasta(bed_files)
    pass
