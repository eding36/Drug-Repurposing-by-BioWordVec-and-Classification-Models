import pandas as pd
import requests as rq
import os
import json
import time 
parent_dir =os.path.dirname(os.getcwd()) 

drugmech_array = pd.read_csv(os.path.join(parent_dir,'DrugmechDB Data/Processed Data/DrugMechDB Processed Triples.csv'))

robokops_array = pd.read_csv(os.path.join(parent_dir,'ROBOKOP Data/ROBOKOP Processed Triples.csv'))

stacked_df = pd.concat([drugmech_array, robokops_array], axis=0,ignore_index=True)


"""Node normalizer function which utilizes the RENCI node norm tool. outputs normalized id and name for an entity."""
def run_node_normalizer(id):
    id_url = id.replace(":","%3A")
    URL= f"https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes?curie={id_url}&conflate=true&drug_chemical_conflate=true&description=false"
    response = rq.get(url = URL)
    response_json = response.json()
    if response.status_code == 200 and response_json[id] != None:
        identifier = response_json[id]["id"]["identifier"]
        name = response_json[id]["id"]["label"].lower()
    return identifier, name

def normalize_entity(entity_id_col, entity_name_col, entity_dict):
    for index, row in stacked_df.iterrows():
        try:
            print(f"{entity_name_col} Iteration: {index}")
            entity_id = row[entity_id_col]
            # Check if the entity ID is already in the dictionary
            if entity_id in entity_dict:
                stacked_df.loc[index, entity_name_col] = entity_dict[entity_id][1]
                stacked_df.loc[index, entity_id_col] = entity_dict[entity_id][0]
            else:
                # Normalize and update the dictionary
                norm_id, norm_name = run_node_normalizer(entity_id)
                entity_dict[entity_id] = [norm_id, norm_name]
                stacked_df.loc[index, entity_name_col] = norm_name
                stacked_df.loc[index, entity_id_col] = norm_id
        except UnboundLocalError:
            stacked_df.loc[index, entity_name_col] = 0
            print(f"Skipped {index} because node normalizer can't find this")
            continue

# Initialize dictionaries
drug_dict = {}
disease_dict = {}
gene_dict = {}

# Normalize drugs
normalize_entity('drug_id', 'drug_name', drug_dict)

# Normalize diseases
normalize_entity('disease_id', 'disease_name', disease_dict)

# Normalize genes
normalize_entity('gene_id', 'gene_name', gene_dict)

"""Data cleaning: remove all nodes that can't be node normalized"""

columns_to_check = ['drug_name','disease_name','gene_name']
stacked_df = stacked_df[~(stacked_df[columns_to_check] == 0).any(axis=1)] 

"""Using HGNC complete set to map gene names to their normalized geneProduct names. Remove those that can't be mapped"""
#####(IF YOU ONLY WANT GENE NAMES SKIP THIS STEP)######
def GetProteinName(gene_name):
    try:
        i = hgnc_df[hgnc_df['symbol']==gene_name.upper()].index.values
        #print(str(i) + " is the index")
        index = int(i[0])
        protein = hgnc_df.at[index, 'name']
        #print(gene + " maps to " + protein)
    except:
        print(f"Could not map gene symbol:{gene_name}")
        protein = 'null'
    return protein

for i in range(len(stacked_df)):
    protein_descriptor = GetProteinName(stacked_df.iloc[i,4]).lower()
    stacked_df.iloc[i,4] = protein_descriptor

json_file = os.path.join(parent_dir, 'hgnc_complete_set.json')
with open(json_file, 'r') as file:
    data = json.load(file)

data_len = len(data['response']['docs'])

gene_prot_list = []

for i in range(data_len):
    gene_symbol = data['response']['docs'][i]['symbol']
    protein_name = data['response']['docs'][i]['name']
    row = [gene_symbol,protein_name]
    gene_prot_list.append(row)

columns = ['symbol','name']

hgnc_df = pd.DataFrame(gene_prot_list, columns=['symbol','name'])

for i in range(len(stacked_df)):
    protein_descriptor = GetProteinName(stacked_df.iloc[i,4]).lower()
    stacked_df.iloc[i,4] = protein_descriptor

mask = stacked_df['gene_name'] != 'null' ###remove all rows where gene couldnt be converted to protein name
filtered_df = stacked_df[mask]
filtered_df.rename(columns ={'gene_name':'protein_name'},inplace = True) ####we only select drug, disease, and protein columns for final dataframe
final_columns = ['drug_name','disease_name','protein_name']
final_df = filtered_df.drop_duplicates(subset = final_columns)
final_df = final_df.reset_index().drop('index', axis = 1).drop('gene_id',axis=1).drop('drug_id',axis=1).drop('disease_id',axis=1)

final_df.to_csv(os.path.join(parent_dir,'ROBOKOP+DrugmechDB Data/ROBOMechDB Processed Triples.csv'), index = False)