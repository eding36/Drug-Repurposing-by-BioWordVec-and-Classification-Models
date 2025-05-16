###FUNCTIONS#####
from functions import csvtodict, compute_threshold, generate_restricted_negatives, build_model
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


model_auc_values = []

unique_protein_vectors = np.array(protein_df.iloc[:,1:201])
unique_disease_vectors = np.array(disease_df.iloc[:,1:201])
unique_drug_vectors = np.array(drug_df.iloc[:,1:201])

unique_protein_dict_names = np.array(protein_df.iloc[:,0])
unique_disease_dict_names = np.array(disease_df.iloc[:,0])
unique_drug_dict_names = np.array(drug_df.iloc[:,0])

drug_threshold = compute_threshold(unique_drug_vectors)
disease_threshold = compute_threshold(unique_disease_vectors)
protein_threshold = compute_threshold(unique_protein_vectors)

for i in random_seeds:
    random.seed(i)
    random_rows = positive_triples_dataframe.sample(int(len(positive_triples_dataframe)/4), random_state = i)
    #we are now going to split these rows into 3 equal chunks. 
    #the triples in the first chunk will have its protein parameter randomized, 
    #the triples in the second chunk will have its disease parameter randomized, and so on

    temp_set = set()

# Split the data
    split_rows = np.array_split(random_rows, 3)
    drug_disease_x, drug_x_protein, x_disease_protein = split_rows
    

# Generate restrictedNegative triples for each category
    drug_disease_x_negatives = generate_restricted_negatives(
    drug_disease_x, (403, 603), unique_protein_vectors, unique_protein_dict_names, 2,
    protein_threshold, pos_neg_triples_dictionary, temp_set, max_count=6, portion = 1
    )

    drug_x_protein_negatives = generate_restricted_negatives(
    drug_x_protein, (203, 403), unique_disease_vectors, unique_disease_dict_names, 1,
    disease_threshold, pos_neg_triples_dictionary, temp_set, max_count=6, portion = 1
    )

    x_disease_protein_negatives = generate_restricted_negatives(
    x_disease_protein, (3, 203), unique_drug_vectors, unique_drug_dict_names, 0,
    drug_threshold, pos_neg_triples_dictionary, temp_set, max_count=5, portion = 1
    )

# Combine all negative triples and create the final dataframe
    negative_df = pd.concat([pd.DataFrame(triples) for triples in [drug_disease_x_negatives,    drug_x_protein_negatives, x_disease_protein_negatives]], ignore_index=True)
    df = pd.concat([positive_triples_dataframe, negative_df], axis=0, ignore_index=True)
    df.columns = unified_columns
    print(df)

    ##check to make sure no duplicates
    test = pd.concat([df_diverseNegs, df],axis=0)
    col = ['0', '1', '2']
    print(len(test))
    rem = test.drop_duplicates(subset = col)
    print(len(rem))
    print(i)
    
    # Perform 5-fold cross-validation to test the accuracy of the model######
    stat_list = build_model(df, i, 2)
    model_auc_values.append(stat_list)
    
fcv_df = pd.DataFrame(model_auc_values, columns=['min_auc', 'max_auc', 'fcv_mean_auc','fcv_std_auc'])

min_auc = np.min(fcv_df.iloc[:,0])
max_auc = np.max(fcv_df.iloc[:,1])
avg_auc = np.mean(fcv_df.iloc[:,2])
std_dev_auc = np.mean(fcv_df.iloc[:,3])

ensemble_stats = np.array([min_auc,max_auc,avg_auc,std_dev_auc])
ensemble_stats = ensemble_stats.reshape(1, -1)
ensemble_stats_df = pd.DataFrame(ensemble_stats, columns = ['min_auc', 'max_auc', 'avg_auc', 'std_dev_auc'])
ensemble_stats_df.to_csv(os.path.join(os.getcwd(), 'Classification Models/Models Trained on restrictedNegatives/restrictedNegatives_models_ensemble_stats.csv'))

