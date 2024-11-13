import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve
import tensorflow as tf
import os

def csvtodict(filename):
    vector_dict = {}
    df_vector_dict = pd.read_csv(filename, header = 0)
    keys = df_vector_dict.iloc[:,0].tolist()
    values = list(range(len(df_vector_dict)))
    for key,value in zip(keys,values):
        vector_dict[key] = value
    return vector_dict


def compute_threshold(vector_dict, n=0.5):
    distances = []
    for i in range((len(vector_dict) - 1)):
        smallest_distance = 10000000
        for j in range(i + 1, len(vector_dict)):
            distance = np.linalg.norm(vector_dict[i] - vector_dict[j])
            if distance < smallest_distance:
                smallest_distance = distance
        distances.append(smallest_distance)

    distances = np.array(distances)

    mean_distances = np.sum(distances) / len(distances)
    stdev_distance = np.std(distances)
    applicability_domain = mean_distances + (n * stdev_distance)

    print('min distance:', np.min(distances))
    print('mean distance:', mean_distances)
    print('stdev distances:', stdev_distance)

    return applicability_domain


def generate_restricted_negatives(reference_df, vector_indices, unique_vectors, unique_names, name_indices, threshold,
                                  pos_triples_dict, temp_set, max_count, portion):
    negative_triples = []
    amount_to_iterate = int(len(reference_df) / portion)
    for i, row in reference_df.iloc[:amount_to_iterate].iterrows():
        reference_vector = np.array(row[vector_indices[0]:vector_indices[1]], dtype=float)
        counter = 0

        for j, unique_vector in enumerate(unique_vectors):

            if np.linalg.norm(reference_vector - unique_vector) < threshold:
                triple_name = [row[0], row[1], row[2]]
                triple_name[name_indices] = unique_names[j]
                new_combo = f"{triple_name[0]} {triple_name[1]} {triple_name[2]}"

                if new_combo in pos_triples_dict or new_combo in temp_set:
                    continue

                temp_set.add(new_combo)
                triple_vector = list(itertools.chain(
                    *[np.array(row[3:vector_indices[0]]).tolist(), unique_vector.tolist(),
                      np.array(row[vector_indices[1]:603]).tolist()]
                ))
                negative_triples.append(list(itertools.chain(*[triple_name, triple_vector, [0]])))

                counter += 1
                if counter >= max_count:
                    break

    return negative_triples

def build_model(df, i, complete_set_id):
    ###SHUFFLE MODEL TRAINING SET TO MIX POSITIVE AND NEGATIVE CASES
    tf.keras.utils.set_random_seed(i)
    shuffled_df = df.sample(frac=1.0, random_state=i)

   # print(shuffled_df)
    data = shuffled_df.iloc[:, 3:-1].values
    labels = shuffled_df.iloc[:, -1].values

    # Perform 5-fold cross-validation to test the accuracy of the model
    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=i)
    all_predictions = []
    all_test_sets = []
    fold = 1
    auc_values = []

    for train_index, test_index in skf.split(data, labels):
        print(f"Fold: {fold}")

        # Split the data into training and test sets
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        tf_train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        tf_test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        BATCH_SIZE = 32
        STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
        tf_train_set = tf_train_set.repeat().batch(BATCH_SIZE)
        tf_test_set = tf_test_set.batch(BATCH_SIZE)

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH * 1000,
            decay_rate=1,
            staircase=False
        )

        def get_optimizer():
            return tf.keras.optimizers.Adam(lr_schedule)

        optimizer = get_optimizer()

        if complete_set_id == 1:
            NUM_EPOCHS = 40
            MODEL_SAVE_PATH = os.getcwd() + f"/Classification Models/Complete Set 1 Models/ROBOMechDB Complete Set 1 Model Seed {i}.keras"
            model = tf.keras.models.Sequential([
                tf.keras.Input(shape=(600,)),
                tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.8),
                tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.8),
                tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.8),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            NUM_EPOCHS = 100
            MODEL_SAVE_PATH = os.getcwd() + f"/Classification Models/Complete Set 2 Models/ROBOMechDB Complete Set 2 Model Seed {i}.keras"

            model = tf.keras.models.Sequential([
                tf.keras.Input(shape=(600,)),
                tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                          restore_best_weights=True)
        # Train the model
        model.fit(tf_train_set,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=NUM_EPOCHS,
                    validation_data=tf_test_set,
                    callbacks=[early_stopping],
                    verbose=0)

        predictions = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        print("AUROC for test data:", roc_auc)
        all_predictions.append(predictions)
        all_test_sets.append(y_test)
        save_model = tf.keras.models.save_model(model, MODEL_SAVE_PATH)

        fold += 1

    fcv_mean_auc = np.mean(auc_values)
    fcv_std_auc = np.std(auc_values)
    fcv_min_auc = np.min(auc_values)
    fcv_max_auc = np.max(auc_values)
    return [fcv_min_auc, fcv_max_auc, fcv_mean_auc, fcv_std_auc]