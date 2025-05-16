###FUNCTIONS#####
from functions import csvtodict, build_model
import pandas as pd
import numpy as np
import os
import itertools
import random


parent_dir = os.path.dirname(os.getcwd())
print(parent_dir)

unified_columns = [f"{i}" for i in range(604)]
filename = os.path.join(parent_dir, 'Step 1 Data Processing/ROBOKOP+DrugMechDB/ROBOKOP+DrugmechDB Data/ROBOMechDB Processed Triples.csv') ###read in the therapeutic triples
df=pd.read_csv(filename)

df.columns = ['0','1','2']

df_80 = df.sample(n=int(0.8*len(df)), random_state=42)  ##take 80% of the positive therapeutic triples
df_80_indices = df_80.index.tolist()

####Find all unique drugs, diseases, and proteins in this 80% set. 
unique_triples_drug = sorted(list(set(df_80['0'])))
unique_triples_disease =sorted(list(set(df_80['1'])))
unique_triples_protein = sorted(list(set(df_80['2'])))



###Create a dictionary with all the external validation set triples so they're not accidentally part of model training set
df_randNegs = pd.read_csv(os.path.join(parent_dir, 'Step 3 External Validation and Model Development/External Validation Datasets/randomNegatives External Validation Set.csv'))
df_restrictedNegs = pd.read_csv(os.path.join(parent_dir, 'Step 3 External Validation and Model Development/External Validation Datasets/restrictedNegatives External Validation Set.csv'))
df_diverseNegs = pd.read_csv(os.path.join(parent_dir, 'Step 3 External Validation and Model Development/External Validation Datasets/diverseNegatives External Validation Set.csv'))

df_ext = pd.concat([df_randNegs.iloc[:,:3],df_restrictedNegs.iloc[:,:3],df_diverseNegs.iloc[:,:3],df],axis=0)
pos_neg_triples_dictionary = {}
values = my_list = [1] * len(df_ext)
keys = [df_ext.iloc[i,0] + " " + df_ext.iloc[i,1]+ " " + df_ext.iloc[i,2] for i in range(0,len(df_ext))]

for key,value in zip(keys,values):
    pos_neg_triples_dictionary[key] = value


"""Importing created protein, disease, and drug vector dictionaries into here"""
protein_dict = csvtodict(os.path.join(parent_dir,
                                      'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Protein Vector Dictionary.csv'))

disease_dict = csvtodict(os.path.join(parent_dir,
                                      'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Disease Vector Dictionary.csv'))

drug_dict = csvtodict(os.path.join(parent_dir,
                                   'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Drug Vector Dictionary.csv'))

protein_df = pd.read_csv(os.path.join(parent_dir,
                                      'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Protein Vector Dictionary.csv'),
                         header=0)
disease_df = pd.read_csv(os.path.join(parent_dir,
                                      'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Disease Vector Dictionary.csv'),
                         header=0)
drug_df = pd.read_csv(os.path.join(parent_dir,
                                   'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Drug Vector Dictionary.csv'),
                      header=0)

"""Create feature embedding dataframe for positive triples"""
positive_triples_vector_array = []
for i in range(0,len(df_80)):
    try:
        drug_vector = drug_df.iloc[drug_dict[df_80.iloc[i,0]],1:201].tolist()
        disease_vector = disease_df.iloc[disease_dict[df_80.iloc[i,1]],1:201].tolist()
        protein_vector = protein_df.iloc[protein_dict[df_80.iloc[i,2]],1:201].tolist()
        
        drug_name = df_80.iloc[i,0]
        disease_name = df_80.iloc[i,1]
        protein_name = df_80.iloc[i,2]
        
        row = [[drug_name],[disease_name],[protein_name],drug_vector,disease_vector,protein_vector,[1]]
        merged = list(itertools.chain(*row))
        positive_triples_vector_array.append(merged)

    except KeyError:
        continue

positive_triples_dataframe = pd.DataFrame(positive_triples_vector_array)

###Initialize random seeds (so we can replicate experiments)
random_seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,42]

model_auc_values = []

"""Training Set Development for Models Trained on randomNegatives"""

for i in random_seeds:
    random.seed(i)

    negative_triples_array = []
    negative_triples_vector_array = []
    temp = set()
    j=0
    while j < int(1.1*len(positive_triples_dataframe)):  ###create negative triples
        drug = random.sample(unique_triples_drug,k=1)[0]
        protein = random.sample(unique_triples_protein, k=1)[0]
        disease = random.sample(unique_triples_disease,k=1)[0]
        if (drug + " " + disease + " " + protein) in pos_neg_triples_dictionary or (drug + " " + disease + " " + protein) in temp:
            continue
        temp.add(drug + " " + disease + " " + protein)
        negative_triples_array.append([drug, disease, protein])
        j+= 1 

    df_negative_triples = pd.DataFrame(negative_triples_array)
    j=0
    for j in range(0,len(df_negative_triples)):
        try:
            drug_vector = drug_df.iloc[drug_dict[df_negative_triples.iloc[j,0]],1:201].tolist()
            disease_vector = disease_df.iloc[disease_dict[df_negative_triples.iloc[j,1]],1:201].tolist()
            protein_vector = protein_df.iloc[protein_dict[df_negative_triples.iloc[j,2]],1:201].tolist()
        
            drug_name = df_negative_triples.iloc[j,0]
            disease_name = df_negative_triples.iloc[j,1]
            protein_name = df_negative_triples.iloc[j,2]
        
            row = [[drug_name],[disease_name],[protein_name],drug_vector,disease_vector,protein_vector,[0]]
            merged = list(itertools.chain(*row))
            negative_triples_vector_array.append(merged)

        except KeyError:
            continue

    negative_triples_dataframe = pd.DataFrame(negative_triples_vector_array)
    df = pd.concat([positive_triples_dataframe,negative_triples_dataframe],axis=0,ignore_index=True)
    df.columns = unified_columns
    
    ###CHECK TO MAKE SURE NO OVRLAP BETWEEN TRAIN AND EXTERNAL VALIDATION SET
    test = pd.concat([df_diverseNegs, df],axis=0)
    col = ['0', '1', '2']
    print(len(test))
    rem = test.drop_duplicates(subset = col)
    print(len(rem))
    ##Build 20 models generate performance stats
    stat_list = build_model(df, i, 1)
    
    model_auc_values.append(stat_list)

fcv_df = pd.DataFrame(model_auc_values, columns=['min_auc', 'max_auc', 'fcv_mean_auc','fcv_std_auc'])

min_auc = np.min(fcv_df.iloc[:,0])
max_auc = np.max(fcv_df.iloc[:,1])
avg_auc = np.mean(fcv_df.iloc[:,2])
std_dev_auc = np.mean(fcv_df.iloc[:,3])

ensemble_stats = np.array([min_auc,max_auc,avg_auc,std_dev_auc])
ensemble_stats = ensemble_stats.reshape(1, -1)
ensemble_stats_df = pd.DataFrame(ensemble_stats, columns = ['min_auc', 'max_auc', 'avg_auc', 'std_dev_auc'])
ensemble_stats_df.to_csv(os.path.join(os.getcwd(), 'Classification Models/Models Trained on randomNegatives/randomNegatives_models_ensemble_stats.csv'))
