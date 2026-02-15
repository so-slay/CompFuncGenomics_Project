#!/usr/bin/env bash

set -euo pipefail

# Fetch FASTA Sequences from BED file

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <input.bed> <output.fa> <ref_genome.fa>"
  exit 1
fi

input_fa="$3"
output_fa="$2"

echo "bedtools version:"
bedtools --version

echo "Used hg38 ref genome from: "$input_fa" "

bedtools getfasta -fo "$output_fa" -fi "$input_fa" -bed "$1"
