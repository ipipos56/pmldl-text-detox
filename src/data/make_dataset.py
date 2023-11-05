import os
import re

# Path to the input file and output file
input_file_path = "filtered.tsv"
output_file_path = "train.txt"


def read_txt(file_path):
    with open(file_path, "r", encoding='latin') as file:
        text = file.read()
    return text


def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text



# Read the input file and write the formatted data to the output file
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    # Skip the header
    infile.readline()

    for line in infile:
        # Split the line by tabs
        fields = line.strip().split('\t')

        # Extract the "reference" and "translation" fields
        reference, translation = fields[1], fields[2]

        # Write the formatted data to the output file
        outfile.write(f"[Q] {reference}\n[A] {translation}\n")

