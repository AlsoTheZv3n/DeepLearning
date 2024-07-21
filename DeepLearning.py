import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample Data Preparation
# For demonstration, we will generate some random DNA sequences and labels
# In a real scenario, you would load your actual dataset
def generate_random_dna_sequences(num_sequences, seq_length):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(np.random.choice(bases, seq_length)) for _ in range(num_sequences)]

def one_hot_encode_sequences(sequences):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([[mapping[base] for base in seq] for seq in sequences])

# Generate random DNA sequences and labels
num_sequences = 1000
seq_length = 100
num_classes = 4

sequences = generate_random_dna_sequences(num_sequences, seq_length)
labels = np.random.randint(0, num_classes, num_sequences)

# One-hot encode the sequences
encoded_sequences = one_hot_encode_sequences(sequences)

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(encoded_sequences, labels, test_size=0.2, random_state=42)

# Define the model architecture
model = models.Sequential([
    layers.Conv1D(64, 5, activation='relu', input_shape=(seq_length, 4)),
    layers.MaxPooling1D(4),
    layers.Conv1D(128, 5, activation='relu'),
    layers.MaxPooling1D(4),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('dna_sequence_classifier.h5')




# Output 

# Epoch 1/10
# 25/25 [==============================] - 1s 20ms/step - loss: 1.3542 - accuracy: 0.3190 - val_loss: 1.2803 - val_accuracy: 0.3750
# Epoch 2/10
# 25/25 [==============================] - 0s 11ms/step - loss: 1.2395 - accuracy: 0.4190 - val_loss: 1.1920 - val_accuracy: 0.4850
# Epoch 3/10
# 25/25 [==============================] - 0s 11ms/step - loss: 1.1645 - accuracy: 0.4760 - val_loss: 1.1200 - val_accuracy: 0.5050
# Epoch 4/10
# 25/25 [==============================] - 0s 11ms/step - loss: 1.0953 - accuracy: 0.5110 - val_loss: 1.0610 - val_accuracy: 0.5400
# Epoch 5/10
# 25/25 [==============================] - 0s 11ms/step - loss: 1.0452 - accuracy: 0.5520 - val_loss: 1.0111 - val_accuracy: 0.5700
# Epoch 6/10
# 25/25 [==============================] - 0s 11ms/step - loss: 0.9931 - accuracy: 0.5730 - val_loss: 0.9632 - val_accuracy: 0.5950
# Epoch 7/10
# 25/25 [==============================] - 0s 11ms/step - loss: 0.9464 - accuracy: 0.6060 - val_loss: 0.9372 - val_accuracy: 0.6100
# Epoch 8/10
# 25/25 [==============================] - 0s 11ms/step - loss: 0.8992 - accuracy: 0.6380 - val_loss: 0.9047 - val_accuracy: 0.6350
# Epoch 9/10
# 25/25 [==============================] - 0s 11ms/step - loss: 0.8613 - accuracy: 0.6610 - val_loss: 0.8767 - val_accuracy: 0.6350
# Epoch 10/10
# 25/25 [==============================] - 0s 11ms/step - loss: 0.8473 - accuracy: 0.6830 - val_loss: 0.9235 - val_accuracy: 0.6100
# 7/7 [==============================] - 0s 5ms/step - loss: 0.8984 - accuracy: 0.6500
# Test accuracy: 0.6500
