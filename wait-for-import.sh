#!/bin/bash
set -e  # exit on error

# check if the file path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <file_path>"
    exit 1
fi

FILE_PATH="$1"
MESSAGE_SHOWN=false

# wait until the sync file exists
until [ -f "$FILE_PATH" ]; do
    if [ "$MESSAGE_SHOWN" = false ]; then
        echo "Waiting for Neo4j import to complete..."
        MESSAGE_SHOWN=true
    fi
    sleep 5
done

echo "Detected import done"
# delete the sync file
rm -f "$FILE_PATH"