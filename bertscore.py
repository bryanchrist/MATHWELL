from evaluate import load
import pandas as pd
import numpy as np 
bertscore = load("bertscore")

egsm = pd.read_csv('data/egsm.csv')

# Note this function expects two pandas dfs as input, column names for the questions in each df as strings, and optional arguments for whether 
# you are calculating within df bertscore and the limit for how many rows you want to compare. The function itself calculates bertscore for all pairwise comparisons
# within the specified limit. 
def score(df1, df2, df1var, df2var, same_df = False, limit = 2000):
    # Create lists to store metrics
    precision = []
    recall = []
    f1 = []
        
    if same_df == False:
        # Shuffle dataframes
        df1 = df1.sample(frac = 1)
        df2 = df2.sample(frac = 1)

        # Dynamically adjust limit based on sizes of DFs
        if limit > len(df1) or limit > len(df2):
            lim1 = min(limit, len(df1))
            lim2 = min(limit, len(df2))
            
        else: 
            lim1 = limit
            lim2 = limit


        # Create lists for references and predictions for bertscore
        refs = []
        preds = []

        # Loop over all possible combinations of predictions and references, computing bertscore when lists hit a len of 128
        for i in range(0, lim1):
            for j in range(0, lim2):
                ref = df1.iloc[i][df1var]
                ref = str(ref)
                pred = df2.iloc[j][df2var]
                pred = str(pred)
                preds.append(pred)
                refs.append(ref)
                if len(preds)==128:
                    results = bertscore.compute(predictions=preds, references=refs, lang="en", batch_size = 128)
                    precision.append(np.mean(results['precision']))
                    recall.append(np.mean(results['recall']))
                    f1.append(np.mean(results['f1']))
                    refs = []
                    preds = []

    # This code is the same as above, except that when you are calculating a within df bertscore, it starts by comparing references and predictions from opposite ends of the df to 
    # minimize overlap 
    if same_df == True:
        if limit > len(df1) or limit > len(df2):
            lim1 = min(limit, len(df1))
            lim2 = min(limit, len(df2))
            
        else: 
            lim1 = limit
            lim2 = limit
            
        refs = []
        preds = []
        for i in range(0, lim1):
            for j in range(len(df2)-lim2, len(df2)):
                ref = df1.iloc[i][df1var]
                ref = str(ref)
                pred = df2.iloc[j][df2var]
                pred = str(pred)
                preds.append(pred)
                refs.append(ref)
                if len(preds)==128:
                    results = bertscore.compute(predictions=preds, references=refs, lang="en", batch_size = 128)
                    precision.append(np.mean(results['precision']))
                    recall.append(np.mean(results['recall']))
                    f1.append(np.mean(results['f1']))
                    refs = []
                    preds = []
                    
    return (precision, recall, f1)

# Example usage
scores = score(egsm, egsm, 'question', 'question')
result = f"Average EGSM overall BERTScore: Precision: {np.mean(scores[0])}, Recall: {np.mean(scores[1])}, F1: {np.mean(scores[2])}"
print(result) 
