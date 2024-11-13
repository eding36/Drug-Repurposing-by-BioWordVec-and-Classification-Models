###README####

FOLLOW THESE STEPS TO CREATE THE ROBOMECHDB DATASET:

1. First, to parse DrugMechDB raw data (indication_paths), open the DrugMechDB Parsing Code folder and run the notebook.

2. Then, to extract ROBOKOP triples using cypher query, open the ROBOKOP Parsing Code folder and run the notebook.

3. Finally, to combine both datasets into one, deduplicate, and normalize all drug, disease, and protein names, open the ROBOMechDB Node Normalizer folder and run the notebook.

TO PROCESS ROBOKOP SCREENING SET TRIPLES:
run "format robokop text mined triples.ipynb"