import tensorflow as tf
import pandas as pd
import numpy as np
import os


parent_dir = os.path.dirname(os.getcwd())


model_directory = os.path.join(parent_dir, 'Step 3 External Validation and Model Development/Classification Models')

model_1_list = []
model_2_list = []
#,8,9,10,11,12,13,14,15,16,17,18,41
# Assuming your models are named 'model_0.h5', 'model_1.h5', ..., 'model_19.h5'
model1_paths = [os.path.join(model_directory, f'Models Trained on randomNegatives/ROBOMechDB randomNegatives Model Seed {i+1}.keras') for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,41]]

model2_paths = [os.path.join(model_directory, f'Models Trained on restrictedNegatives/ROBOMechDB resrictedNegatives Model Seed {i+1}.keras') for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,41]]


for path in model1_paths:
    print(f'Loading model from {path}')
    model = tf.keras.models.load_model(path)
    model_1_list.append(model)
    print(f'Model from {path} loaded successfully')

for path in model2_paths:
    print(f'Loading model from {path}')
    model = tf.keras.models.load_model(path)
    model_2_list.append(model)
    print(f'Model from {path} loaded successfully')


#loading triple embeddings from robokops kg
df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'Step 2 Data Embedding/Data Mined Embedded Dataset/ROBOKOP Virtual Screening Set Embedded Dataset.csv'))
df = df.drop('Unnamed: 0', axis = 1)
df = df.drop_duplicates(subset = ['0','2','4'])

data = df.iloc[:, 6:-1]

model_1_prediction_matrix = []
model_2_prediction_matrix = []

for j in range(len(model_1_list)):
    model_1_prediction_value = model_1_list[j].predict(data)
    model_1_prediction_value = model_1_prediction_value.flatten()
    model_1_prediction_matrix.append(model_1_prediction_value)
    
    model_2_prediction_value = model_2_list[j].predict(data)
    model_2_prediction_value = model_2_prediction_value.flatten()
    model_2_prediction_matrix.append(model_2_prediction_value)

model_1_prediction_matrix = np.array(model_1_prediction_matrix)
model_2_prediction_matrix = np.array(model_2_prediction_matrix)
    
model_1_stability_matrix = model_1_prediction_matrix.T
model_2_stability_matrix = model_2_prediction_matrix.T

model_1_prediction_floats = np.array([np.mean(row) for row in model_1_stability_matrix])
model_2_prediction_floats = np.array([np.mean(row) for row in model_2_stability_matrix])
model_prediction_floats = 0.9*model_1_prediction_floats + 0.1*model_2_prediction_floats

data_rounded = np.round(model_prediction_floats)
descending_confidence_values = np.sort(model_prediction_floats.flatten())[::-1]
ascending_indices = np.argsort(model_prediction_floats.flatten())
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
    conf_value = float(model_prediction_floats[i])
    row = [triple_name,conf_value]
    flat_row = row[0]+[row[1]]
    prediction_array.append(flat_row)

columns_to_use = ['drug_name','drug_id','disease_name','disease_id','protein_name','protein_id','model_confidence_value']
df = pd.DataFrame(data = prediction_array,columns = columns_to_use)
print(df)
df.to_csv(os.path.join(os.getcwd(),'Screening Results/Bag of Ensembles Predictions of Top ROBOKOP Virtual Screening Set Triples.csv'))

df_mined = pd.read_csv(os.path.join(os.getcwd(),'Screening Results/Bag of Ensembles Predictions of Top ROBOKOP Virtual Screening Set Triples.csv'))
df_mined.drop(['Unnamed: 0'], axis = 1, inplace = True)

