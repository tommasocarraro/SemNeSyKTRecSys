#!/bin/bash

for file in *; do
    if [[ -f "$file" && "$file" == *.json ]]; then
        7z a -t7z -m0=lzma2 -mmt=12 -mx=9 -mfb=273 -md=256m "${file}.7z" "$file"
    fi
done
