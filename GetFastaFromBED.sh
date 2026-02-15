#!/usr/bin/env bash

set -euo pipefail

# Fetch FASTA Sequences from BED file

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <input.bed> <output.bed>"
  exit 1
fi

input_fa="/home/s_vanantha/Desktop/IISERpBin/IISERP_Sem7/Sem7_LeelavatiNarlikar/Sem7Proj/hg38/hg38.fa"
output_fa="$2"

echo "bedtools version:"
bedtools --version

echo "Used hg38 ref genome from: "$input_fa" "

bedtools getfasta -fo "$output_fa" -fi "$input_fa" -bed "$1"
