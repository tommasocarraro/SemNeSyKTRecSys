#!/bin/bash

n_cores=$(nproc --all)

for file in *; do
    if [[ -f "$file" && "$file" == *.jsonl ]]; then
        7z a -t7z -m0=lzma2 -mmt="${n_cores}" -mx=9 -mfb=273 -md=256m -v85m "${file}.7z" "$file"
    fi
done
