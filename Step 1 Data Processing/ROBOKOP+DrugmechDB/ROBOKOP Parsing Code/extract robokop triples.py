from neo4j import GraphDatabase
from py2neo import Graph
import pandas as pd

"""PYTHON FRAMEWORK TO RUN NEO4J CYPHER QUERY ON ROBOKOP DATABASE"""

import requests
import json

url = "https://robokop-automat.apps.renci.org/robokopkg/cypher"

payload = json.dumps({"query": """MATCH (c:`biolink:ChemicalEntity`)-[r0:`biolink:binds`|`biolink:directly_physically_interacts_with`]-(g:`biolink:Gene`)-[r1]-(d:`biolink:Disease`),(c)-[r2:`biolink:treats`]-(d) WHERE (properties(c)["CHEBI_ROLE_pharmaceutical"]) IS NOT NULL AND properties(r2)["primary_knowledge_source"]="infores:drugcentral" RETURN DISTINCT c.name,c.id,d.name,d.id,g.name,g.id"""
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

response = requests.post(url, headers=headers, data=payload)
print(response)
response_json = response.json()

values = response_json
num_results = len(values['results'][0]['data'])
triples_list = [item['row'] for item in response_json['results'][0]['data']]

# Convert specified elements to lowercase
for triple in triples_list:
    triple[0] = triple[0].lower()
    triple[2] = triple[2].lower()
    triple[4] = triple[4].lower()

# Print the first four triples
print(triples_list[:4])

import csv
import os
robokop_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'ROBOKOP Data')

df = pd.DataFrame(data = triples_list,columns = ['drug_name','drug_id', 'disease_name','disease_id','gene_name','gene_id'])
df.to_csv(os.path.join(robokop_data_dir, 'ROBOKOP Processed Triples.csv'),quoting=csv.QUOTE_ALL, index=False)