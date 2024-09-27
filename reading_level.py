import textstat
import pandas as pd
import numpy as np

egsm = pd.read_csv('data/egsm.csv')

# This function calculates readability metrics for a pandas dataframe. Note you need to specify the column name for the question text.
def readability(df, varname):
    # Lists to store metrics
    grades = []
    ndcs = []

    # Loop over all questions, calculating and storing scores for each metric
    for i in range(0, len(df)):
        text = df.iloc[i][varname]
        text = str(text)
        grade = textstat.flesch_kincaid_grade(text)
        ndc = textstat.dale_chall_readability_score(text)
        # Reassign grade level under the minimum FKGL score to the minimum score
        if grade < -3.40:
            grade = -3.40
        grades.append(grade)
        ndcs.append(ndc)
    return grades, ndcs

# Example usage 
readability(egsm, 'question')

