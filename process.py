import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import json

def class_choice(df, class_names):
    """Select samples from specific classes."""
    return df[df['dx'].isin(class_names)]

def prepare_labels(labels, model_dir):
        """Convert string labels to one-hot encoding"""
        #create and fit label encoder
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)

        #save label encoder classes so we can use them later for interpretation
        label_mapping = dict(zip(label_encoder.classes_,
                               range(len(label_encoder.classes_))))
        with open(os.path.join(model_dir, 'label_mapping.json'), 'w') as f:
            json.dump(label_mapping, f)

        #one-hot encoding the numeric-encoded classes
        one_hot_labels = tf.keras.utils.to_categorical(numeric_labels)

        #Print mapping for verification
        print("Label mapping:")
        for label, idx in label_mapping.items():
            print(f"{label}: {idx}")

        return one_hot_labels, label_encoder

#CAREFUL use this function with the training set to prevent data leakage
def calculate_class_weights(y_encoded):
    """
    Calculate class weights from original string labels

    Parameters:
    original_labels: array of original string labels

    Returns:
    dict: mapping of numerical indices to weights
    """
    #Use label encoder to get numerical labels
    numerical_labels = np.argmax(y_encoded, axis=1)

    #Calculate weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(numerical_labels),
        y=numerical_labels
    )

    #Create dictionary mapping class indices to weights
    class_weights = dict(zip(range(len(weights)), weights))

    return class_weights