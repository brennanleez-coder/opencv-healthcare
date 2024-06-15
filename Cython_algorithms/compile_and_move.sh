#!/bin/bash

# Navigate to the directory containing the setup.py file
cd ./Cython_algorithms || exit

# Build the Cython extensions in place and capture output
build_output=$(python3 setup.py build_ext --inplace 2>&1)

# Find all .so files in the root directory and move them to the current directory
find ../ -maxdepth 1 -name "*.so" -exec mv {} ./ \;

# Echo the build and move completion message
echo "Build and move completed"

# Highlight warnings and errors
echo "========================= Build Warnings and Errors ========================="

# Extract warnings and errors from the build output
warnings=$(echo "$build_output" | grep -i "warning")
errors=$(echo "$build_output" | grep -i "error")

# Count the number of warnings and errors
num_warnings=$(echo "$warnings" | grep -c -i "warning")
num_errors=$(echo "$errors" | grep -c -i "error")

# Display warnings and errors separately with counts
if [ "$num_warnings" -gt 0 ]; then
    echo "Warnings:"
    echo "$warnings"
    echo "Total warnings: $num_warnings"
else
    echo "No warnings found."
fi

if [ "$num_errors" -gt 0 ]; then
    echo "Errors:"
    echo "$errors"
    echo "Total errors: $num_errors"
    echo "Please check build_output.log for more details."
else
    echo "No errors found."
fi

echo "$build_output" > build_output.log
