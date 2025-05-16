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

###Compute distance thresholds of proteins, drugs, and diseases present
    
unique_protein_vectors = np.array(protein_df.iloc[:,1:201])
unique_disease_vectors = np.array(disease_df.iloc[:,1:201])
unique_drug_vectors = np.array(drug_df.iloc[:,1:201])

unique_protein_dict_names = np.array(protein_df.iloc[:,0])
unique_disease_dict_names = np.array(disease_df.iloc[:,0])
unique_drug_dict_names = np.array(drug_df.iloc[:,0])

drug_threshold = compute_threshold(unique_drug_vectors)
disease_threshold = compute_threshold(unique_disease_vectors)
protein_threshold = compute_threshold(unique_protein_vectors)

print('distanceThreshold for drug list:', drug_threshold)
print('distanceThreshold for disease list:' , disease_threshold)
print('distanceThreshold for protein list:', protein_threshold)




"""generating randomNegative Triples Portion"""

negative_triples_array = []
negative_triples_vector_array = []
temp = set()
i=0

random.seed(42)

while i < 9.5*len(positive_triples_dataframe):  ###create negative triples 
    drug = random.sample(unique_triples_drug,k=1)[0]
    protein = random.sample(unique_triples_protein, k=1)[0]
    disease = random.sample(unique_triples_disease,k=1)[0]
    if (drug + " " + disease + " " + protein) in pos_triples_dictionary or (drug + " " + disease + " " + protein) in temp:
        continue
    temp.add(drug + " " + disease + " " + protein)
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

rand_negative_triples_dataframe = pd.DataFrame(negative_triples_vector_array)


"""GENERATE RESTRICTEDNEGATIVES PORTION"""

#we are now going to split these rows into 3 equal chunks. 
#the triples in the first chunk will have its protein parameter randomized, 
#the triples in the second chunk will have its disease parameter randomized, and so on

split_rows = np.array_split(positive_triples_dataframe, 3)

# Each part is now a separate DataFrame
drug_disease_x, drug_x_protein, x_disease_protein = split_rows[0], split_rows[1], split_rows[2]

drug_disease_x_negative_triples = []
drug_x_protein_negative_triples = []
x_disease_protein_negative_triples = []


temp_set = set()

# Generate negative triples for each part
drug_disease_x_negatives = generate_restricted_negatives(
    drug_disease_x, (403, 603), unique_protein_vectors, unique_protein_dict_names, 2,
    protein_threshold, pos_triples_dictionary, temp_set, max_count=1, portion = 1.3
)

drug_x_protein_negatives = generate_restricted_negatives(
    drug_x_protein, (203, 403), unique_disease_vectors, unique_disease_dict_names, 1,
    disease_threshold, pos_triples_dictionary, temp_set, max_count=2, portion = 1.25
)

x_disease_protein_negatives = generate_restricted_negatives(
    x_disease_protein, (3, 203), unique_drug_vectors, unique_drug_dict_names, 0,
    drug_threshold, pos_triples_dictionary, temp_set, max_count=1, portion = 1.3
)

# Combine and save the DataFrames
restricted_negative_triples_dataframe = pd.concat([
    pd.DataFrame(drug_disease_x_negatives),
    pd.DataFrame(drug_x_protein_negatives),
    pd.DataFrame(x_disease_protein_negatives)
], axis=0, ignore_index=True)

restricted_negative_triples_dataframe.columns = unified_columns
rand_negative_triples_dataframe.columns = unified_columns
positive_triples_dataframe.columns = unified_columns

neg_total = pd.concat([restricted_negative_triples_dataframe,rand_negative_triples_dataframe],axis=0)
neg_total.columns = unified_columns
final_columns = ['0','1','2']
neg_total = neg_total.drop_duplicates(subset = final_columns).reset_index().drop('index', axis = 1)

df_diverseNegs = pd.concat([positive_triples_dataframe,neg_total],axis=0,ignore_index=True)

df_diverseNegs.to_csv(os.path.join(parent_dir, 'Step 3 External Validation and Model Development/External Validation Datasets/diverseNegatives External Validation Set1.csv'), index=False)