from functions import csvtodict, compute_threshold, generate_restricted_negatives
import os
parent_dir = os.path.dirname(os.getcwd())
print(parent_dir)

import pandas as pd
import numpy as np
import random

unified_columns = [f"{i}" for i in range(604)]
filename = os.path.join(parent_dir, 'Step 1 Data Processing/ROBOKOP+DrugMechDB/ROBOKOP+DrugmechDB Data/ROBOMechDB Processed Triples.csv')
df=pd.read_csv(filename)
print(df)

df_80 = df.sample(n=int(0.8*len(df)), random_state=42)  ##take 80% of these positive triples
df_80_indices = df_80.index.tolist()

all_indices = set(df.index) 
df_20_enrichment_indices = list(all_indices - set(df_80_indices))
df_20 = df.iloc[df_20_enrichment_indices] ###pocketing 20% of positive set out for validation

print(df_20)

####Find all unique drugs, diseases, and proteins in this 80% set. 
unique_triples_drug = sorted(list(set(df_20['drug_name'])))
unique_triples_disease = sorted(list(set(df_20['disease_name'])))
unique_triples_protein = sorted(list(set(df_20['protein_name'])))

pos_triples_dictionary = {}
pos_values = my_list = [1] * len(df)
pos_keys = [df.iloc[i,0] + " " + df.iloc[i,1]+ " " + df.iloc[i,2] for i in range(0,len(df))]
for key,value in zip(pos_keys,pos_values):
    pos_triples_dictionary[key] = value

###Importing created protein, disease, and drug vector dictionaries into here
protein_dict = csvtodict(os.path.join(parent_dir, 'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Protein Vector Dictionary.csv'))

disease_dict = csvtodict(os.path.join(parent_dir, 'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Disease Vector Dictionary.csv'))

drug_dict = csvtodict(os.path.join(parent_dir, 'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Drug Vector Dictionary.csv'))

protein_df = pd.read_csv(os.path.join(parent_dir, 'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Protein Vector Dictionary.csv'),header = 0)
disease_df = pd.read_csv(os.path.join(parent_dir, 'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Disease Vector Dictionary.csv'),header = 0)
drug_df = pd.read_csv(os.path.join(parent_dir, 'Step 2 Data Embedding/Vector Dictionaries/ROBOMechDB Drug Vector Dictionary.csv'), header = 0)

###3 Building a 600 column vector array of all positive triples

positive_triples_vector_array = []
import itertools

for i in range(0,len(df_20)):
    try:
        drug_vector = drug_df.iloc[drug_dict[df_20.iloc[i,0]],1:201].tolist()
        disease_vector = disease_df.iloc[disease_dict[df_20.iloc[i,1]],1:201].tolist()
        protein_vector = protein_df.iloc[protein_dict[df_20.iloc[i,2]],1:201].tolist()
        
        drug_name = df_20.iloc[i,0]
        disease_name = df_20.iloc[i,1]
        protein_name = df_20.iloc[i,2]
        
        row = [[drug_name],[disease_name],[protein_name],drug_vector,disease_vector,protein_vector,[1]]
        merged = list(itertools.chain(*row))
        positive_triples_vector_array.append(merged)

    except KeyError:
        continue

positive_triples_dataframe = pd.DataFrame(positive_triples_vector_array)

###Use random combinations of drug, disease, protein present in positive triples to form negatives. Create negative vector array

negative_triples_array = []
negative_triples_vector_array = []
seen = set()
i=0

random.seed(42)

while i < 10.5*len(positive_triples_dataframe):  ###create negative triples
    drug = random.sample(unique_triples_drug,k=1)[0]
    protein = random.sample(unique_triples_protein, k=1)[0]
    disease = random.sample(unique_triples_disease,k=1)[0]
    key = drug + " " + disease + " " + protein
    if (key in pos_triples_dictionary)or (key in seen):
        continue
    seen.add(key)
    negative_triples_array.append([drug, disease, protein])
    i+= 1 

df_negative_triples = pd.DataFrame(negative_triples_array)
    
for i in range(0,len(df_negative_triples)):
    try:
        drug_vector = drug_df.iloc[drug_dict[df_negative_triples.iloc[i,0]],1:201].tolist()
        disease_vector = disease_df.iloc[disease_dict[df_negative_triples.iloc[i,1]],1:201].tolist()
        protein_vector = protein_df.iloc[protein_dict[df_negative_triples.iloc[i,2]],1:201].tolist()
        
        drug_name = df_negative_triples.iloc[i,0]
        disease_name = df_negative_triples.iloc[i,1]
        protein_name = df_negative_triples.iloc[i,2]
        
        row = [[drug_name],[disease_name],[protein_name],drug_vector,disease_vector,protein_vector,[0]]
        merged = list(itertools.chain(*row))
        negative_triples_vector_array.append(merged)

    except KeyError:
        continue

negative_triples_dataframe = pd.DataFrame(negative_triples_vector_array)

df_randomNegatives = pd.concat([positive_triples_dataframe,negative_triples_dataframe],axis=0,ignore_index=True)
df_randomNegatives.columns = unified_columns
print(df_randomNegatives.columns)


df_randomNegatives.to_csv(os.path.join(parent_dir, 'Step 3 External Validation and Model Development/External Validation Datasets/randomNegatives External Validation Set1.csv'), index=False)
