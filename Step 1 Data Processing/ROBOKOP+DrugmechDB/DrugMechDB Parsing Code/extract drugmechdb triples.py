"""JSON TO CSV FUNCTION to turn drugmechdb raw indication paths.json --> indication paths.csv"""

import json
import os
import pandas as pd
import numpy as np

def json_to_csv(json_file, csv_file):
    with open(json_file) as f:
        json_data = json.load(f)
    
    df = pd.json_normalize(json_data)
    df.to_csv(csv_file, index=False)

current_dir = os.getcwd()

# Define the sibling directory name
drugmechdb_data_dir = os.path.join(os.path.dirname(current_dir), 'DrugMechDB Data')

# Example usage
json_file = os.path.join(drugmechdb_data_dir, 'Raw Data', 'indication_paths.json')
csv_file = os.path.join(drugmechdb_data_dir, 'Raw Data', 'indication_paths.csv')
json_to_csv(json_file, csv_file)

import os

# Load the data
df_raw = pd.read_csv(os.path.join(drugmechdb_data_dir, 'Raw Data', 'indication_paths.csv'))
triples_list = []

print(df_raw)

# Helper function to check if comments and related columns are empty, indicating less reliable indication paths
def is_comments_column_empty(row):
    comment_cols = ['comment', 'comments', 'references', 'commments', 'comemnt']
    return all(pd.isnull(row[col]) for col in comment_cols)

# Patterns for drug-protein interaction edges
interaction_patterns = {
    "positively_regulates": "positively regulates",
    "negatively_regulates": "negatively regulates",
    "decreases_activity": "decreases activity of",
    "increases_activity": "increases activity of"
}

# Iterate over rows
for i, row in df_raw.iterrows():
    # Filter out pathways without proteins
    if 'UniProt:' not in row['links'] or not is_comments_column_empty(row):
        continue
    
    # Extract and clean drug and disease information
    drug = row['graph.drug'].lower().replace("'", "")
    drug_id_mesh = row['graph.drug_mesh']
    drug_id_bank = row['graph.drugbank'].replace('DB:DB', 'DRUGBANK:DB')
    disease = row['graph.disease'].lower().replace("'", "")
    disease_id_mesh = row['graph.disease_mesh']

    # Clean links and group them into triples
    links_clean = row['links'].replace("{", "").replace("}", "").replace("'", "").replace("[", "").replace("]", "").split(", ")
    links_grouped = [', '.join(links_clean[i:i+3]) for i in range(0, len(links_clean), 3)]

    # Generate drug-protein interaction patterns
    source_patterns = [
        f"key: {interaction}, source: {drug_id}, target: UniProt:"
        for interaction in interaction_patterns.values()
        for drug_id in [drug_id_mesh, drug_id_bank.replace('DRUGBANK:', 'DB:')]
    ]

    # Check and extract triples for each link
    for link in links_grouped:
        if any(pattern in link for pattern in source_patterns):
            protein_id = link.split(', ')[2].replace("target: UniProt", "UniProtKB")
            triples_temp = [drug, drug_id_bank, disease, disease_id_mesh, 'null', protein_id]
            triples_list.append(triples_temp)

print(triples_list[:4])

triples_list[228][1] = "DRUGBANK:DB08902" ####fixing a raw data entry error.
triples_list[2301][1] = "DRUGBANK:DB02362"
triples_list[2302][1] = "DRUGBANK:DB02362"
triples_list[2303][1] = "DRUGBANK:DB02362"
triples_list[2304][1] = "DRUGBANK:DB02362"

triples_array = np.array(triples_list)
df = pd.DataFrame(data = triples_array)
df.columns = ['drug_name', 'drug_id', 'disease_name','disease_id','gene_name','gene_id']
df.to_csv(os.path.join(drugmechdb_data_dir, 'Processed Data', 'DrugMechDB Processed Triples.csv'), index = False)