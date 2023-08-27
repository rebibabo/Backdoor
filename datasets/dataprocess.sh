#!/bin/bash
# download python.zip from https://huggingface.co/datasets/code_search_net/blob/main/data/python.zip and mv it into ./codesearch
# make sure the file "./codesearch/python.zip" exists
unzip python.zip
pip install more_itertools tree-sitter datasets more_itertools
python preprocess_data.py
# poisoning the training dataset
cd attack
python poison_data.py
# generate the test data for evaluating the backdoor attack
cd ../
python extract_data.py