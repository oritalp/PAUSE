#!/bin/bash

# Using seq for range 1 to 3
for i in $(seq 1 3); do
    echo "Running iteration $i"
    python check.py --input $i
done
