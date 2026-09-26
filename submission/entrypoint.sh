#!/bin/bash

# entrypoint.sh
#
#   This bash script, a variant of which should be included in your submission
# zip file, will be called by the administrative wrapper (ingestion) program. In
# it, you should run your python (or other language) main script to perform
# inference on the input images and generate your output.
#
#   This script will receive three arguments from the administrative wrapper
# (ingestion) program:
#
# 1. input_data (also called "input_images_dir") - The path with the input DICOM
#        images (one per chest radiograph) that your application should perform
#        inference on.
# 2. output (also called "output_dir") - The path of the folder that your
#        application should write output to.  See the challenge instructions for
#        more details on the files that your application should create.
# 3. ingested_program (also called "submission_dir") - this is the folder where
#        your submission zip file contents (including this "entrypoint.sh")
#        reside.
#
# Note that when this "entrypoint.sh" file is called, your current working
# directory will likely >not< be already set to the submission_dir /ingested_program.
# Set it explicitly if it matters for your code.



# Use the inputs as arguments
input_data=$1
output=$2
ingested_program=$3

# Print the inputs
echo "input_data, $input_data"
echo "output, $output"
echo "ingested_program, $ingested_program"

# Your algorithm (example below)
cd $ingested_program
python3 $ingested_program/main.py $input_data $output $ingested_program