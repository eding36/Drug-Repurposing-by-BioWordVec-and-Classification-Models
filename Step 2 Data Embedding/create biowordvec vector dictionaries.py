from gensim.models import KeyedVectors
import os

##make sure you install the correct version of scipy (v1.12 or earlier or else this block wont work)

model = KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), 'Embedding Models', 'BioWordVec Embedding Model.bin'),binary = True)

parent_dir = os.path.dirname(os.getcwd())

#create vector dictionaries for each unique drug, disease, target used in the triples
import pandas as pd
import csv


###Read in Processed Data from dataset

df = pd.read_csv(os.path.join(parent_dir, 'Step 1 Data Processing/ROBOKOP+DrugMechDB/ROBOKOP+DrugmechDB Data/ROBOMechDB Processed Triples.csv')) #input array for embedding
df = df.drop('Unnamed: 0', axis = 1)

triples_drug = sorted(list(set(df['drug_name'].tolist())))
triples_disease = sorted(list(set(df['disease_name'].tolist())))
triples_protein = sorted(list(set(df['protein_name'].tolist())))

print(len(triples_drug))
print(len(triples_disease))
print(len(triples_protein))

#Vector embeddings

def get_phrase_vec(vocablist, phrase):
    words = phrase.replace(",", "")
    words_split = words.split(' ')
    #print(words)
    count = 0
    flag  = 0
    for i in words_split:
        if i in vocablist:
            if count == 0:
                comb_emb = model[i]
            else:
                comb_emb = model[i] + comb_emb
            count = count + 1
        else:
            flag = 1 + flag
            break
    if flag != 0:
        return flag, phrase, "no embedding"
        #print(phrase, "no embedding") 
        flag = flag
    if flag == 0:
        return flag, phrase, list(comb_emb) ##list(comb_emb) holds the 200 dim vector embedding


vocab = model.index_to_key

def get_embeddings(triples, vocab):
    embeddings = []
    embeddable_list = []
    
    for triple in triples:
        fl, ph, str_ = get_phrase_vec(vocab, triple)
        if str_ == "no embedding":
            continue
        str_array = [float(i) for i in str_]
        embeddings.append(str_array)
        embeddable_list.append(ph)
    
    return embeddings, embeddable_list

def create_embedding_dict(embeddable_list, embeddings):
    return {key: value for key, value in zip(embeddable_list, embeddings)}

# Extract embeddings for all drugs, diseases, and proteins
drug_embeddings, embeddable_drug_list = get_embeddings(triples_drug, vocab)
disease_embeddings, embeddable_disease_list = get_embeddings(triples_disease, vocab)
protein_embeddings, embeddable_protein_list = get_embeddings(triples_protein, vocab)

# Map entity name to entity embedding w/ dictionary
drug_vector_dict = create_embedding_dict(embeddable_drug_list, drug_embeddings)
disease_vector_dict = create_embedding_dict(embeddable_disease_list, disease_embeddings)
protein_vector_dict = create_embedding_dict(embeddable_protein_list, protein_embeddings)

#export vector dictionary for use in embedding triples

with open(os.path.join(os.getcwd(), 'Vector Dictionaries', 'ROBOMechDB Disease Vector Dictionary.csv') , 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header
    writer.writerow(['key'] + [f'value_{i+1}' for i in range(200)])
    
    # Write the key and the values
    for key, values in disease_vector_dict.items():
        writer.writerow([key] + values)

    
with open(os.path.join(os.getcwd(), 'Vector Dictionaries', 'ROBOMechDB Drug Vector Dictionary.csv'), 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header
    writer.writerow(['key'] + [f'value_{i+1}' for i in range(200)])
    
    # Write the key and the values
    for key, values in drug_vector_dict.items():
        writer.writerow([key] + values)

with open(os.path.join(os.getcwd(), 'Vector Dictionaries', 'ROBOMechDB Protein Vector Dictionary.csv'), 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header
    writer.writerow(['key'] + [f'value_{i+1}' for i in range(200)])
    
    # Write the key and the values
    for key, values in protein_vector_dict.items():
        writer.writerow([key] + values)


    
