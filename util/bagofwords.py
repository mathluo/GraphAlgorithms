# some scripts for text processing
import pandas as pd 
import re


def PreProcess(filename): 

	

def BagOfWords(filename):
# script for generating bag of words matrix from a tsv file
# each tab separates a document
# stop words and punctuation and capitalization is removed
# loads the entire file in bulk
	text = pd.csv_read(filename,header = 0, delimiter = '\t')
	for line in text:
		for word in 
