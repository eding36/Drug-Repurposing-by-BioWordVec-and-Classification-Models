import os
import pandas as pd 
import json

print(os.getcwd())

df_raw = pd.read_csv(os.path.join(os.getcwd(),'Raw Data/ROBOKOP Drug-Gene-Target Triples Virtual Screening Set.csv'))

#####import hgnc complete set to change gene names that can be changed to protein names

json_file = os.path.join(os.path.dirname(os.getcwd()), 'ROBOKOP+DrugmechDB/hgnc_complete_set.json')
with open(json_file, 'r') as file:
    data = json.load(file)

data_len = len(data['response']['docs'])

gene_prot_list = []

for i in range(data_len):
    try:
        gene_symbol = data['response']['docs'][i]['symbol']
        protein_name = data['response']['docs'][i]['name']
        protein_id = data['response']['docs'][i]['uniprot_ids'][0]
        row = [gene_symbol,protein_name,protein_id]
        gene_prot_list.append(row)
    except KeyError:
        gene_symbol = data['response']['docs'][i]['symbol']
        protein_name = data['response']['docs'][i]['name']
        protein_id = 'null'
        row = [gene_symbol,protein_name,protein_id]
        gene_prot_list.append(row)

columns = ['symbol','name','uniprot id']

hgnc_df = pd.DataFrame(gene_prot_list, columns=columns)

columns = ['c.name','c.id','d.name','d.id', 'g.name','g.id']
df = df_raw[columns]
df.columns = ['drug_name', 'drug.id','disease_name', 'disease.id','gene_name','gene.id']

for i in range(len(df)):
    df.iloc[i,0] = df.iloc[i,0].lower()
    df.iloc[i,2] = df.iloc[i,2].lower()
    df.iloc[i,4] = df.iloc[i,4].lower()

df['gene_name'] = df['gene_name'].replace('pgk 1.00', 'pgk1').replace('npr 2.00', 'npr2').replace('top 1.00', 'top1').replace('cyp 20.00', 'cyp20')

#####TO OBTAIN DATASET WITH ONLY PROTEIN DESCRIPTORS (IF YOU ONLY WANT GENE NAMES SKIP THIS STEP)######

def GetProteinName(gene_name):
    try:
        i = hgnc_df[hgnc_df['symbol']==gene_name.upper()].index.values
        #print(str(i) + " is the index")
        index = int(i[0])
        protein = hgnc_df.at[index, 'name']
        protein_id = hgnc_df.at[index, 'uniprot id']
        #print(gene + " maps to " + protein)
    except:
        print(f"Could not map gene symbol:{gene_name}")
        protein = 'null'
        protein_id = 'null'
    return protein, protein_id

for i in range(len(df)):
    protein_descriptor, protein_id = GetProteinName(df.iloc[i,4])
    df.iloc[i,4] = protein_descriptor.lower()
    df.iloc[i,5] = protein_id

filtered_df = df[(df['gene_name'] != "null") & (df['gene.id'] != "null")]  ###remove all rows where gene couldnt be converted to protein name
filtered_df.rename(columns ={'gene_name':'protein_name', 'gene.id':'protein_id'},inplace = True)
filtered_df.to_csv(os.path.join(os.getcwd(), 'Processed Data/ROBOKOP Virtual Screening Set Triples.csv')
)