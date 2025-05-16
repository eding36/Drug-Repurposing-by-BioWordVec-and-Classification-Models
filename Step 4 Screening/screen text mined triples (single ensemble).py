import tensorflow as tf
import pandas as pd
import os
import numpy as np

parent_dir = os.path.dirname(os.getcwd())
print(parent_dir)

loaded_model = tf.keras.models.load_model(
os.path.join(parent_dir, 'Data Embedding & Model Development/Classification Models/Models Trained on randomNegatives/ROBOMechDB randomNegatives Model Seed 1.keras'))

#loading triple embeddings from robokops kg
df = pd.read_csv(os.path.join(parent_dir,'Data Embedding & Model Development/Data Mined Embedded Dataset/ROBOKOP Virtual Screening Set Embedded Dataset.csv')
)
df = df.drop('Unnamed: 0', axis = 1)
df = df.drop_duplicates(subset = ['0','2','4'])
data = df.iloc[:, 6:-1]
predictions = loaded_model.predict(data)
data_rounded = np.round(predictions)
descending_confidence_values = np.sort(predictions.flatten())[::-1]
ascending_indices = np.argsort(predictions.flatten())
descending_indices = np.flip(ascending_indices)

## finding the indices of only the 1's (actives)

true_positive_indices = []
true_positive_count = 0

false_negative_indices = []
false_negative_count = 0

for i in range (0,len(data_rounded)):
    if data_rounded[i] == 1:
        true_positive_indices.append(i)
        true_positive_count += 1
    else:
        false_negative_indices.append(i)
        false_negative_count +=1

print(true_positive_indices[:20])
print(true_positive_count)
print(false_negative_indices[:20])
print(false_negative_count)

prediction_array = []
for i in descending_indices:
    triple_name = df.iloc[i,:6].tolist()
    conf_value = float(predictions[i])
    row = [triple_name,conf_value]
    flat_row = row[0]+[row[1]]
    prediction_array.append(flat_row)

columns_to_use = ['drug_name','drug_id','disease_name','disease_id','protein_name','protein_id','model_confidence_value']
df = pd.DataFrame(data = prediction_array,columns = columns_to_use)
df = df.drop_duplicates(subset= ['drug_name','disease_name','protein_name'])
df.to_csv(os.path.join(os.getcwd(),'Screening Results/Complete Set 1 Top Model Predictions of Top ROBOKOP Data Mined Triples.csv'))