import tensorflow
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

sentence = ["I am feeling sad today.", 
            "I had a bad day at school. i got hurt while playing football"]

# Tokenization

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentence)

# Create a word_index dictionary

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentence)
print(sequences[0:2])

# Padding the sequence
padded = pad_sequences(sequences, maxlen=100, 
                                padding='post', truncating='post')

print(padded[0:2])

# Define the model using .h5 file
model = tensorflow.keras.models.load_model('C:/Users/HI User/Desktop/WHJR Python/115/sentiment/Scripts/Text_Emotion.h5')

# Test the model
result = model.predict(padded)
print(result)

# Print the result
predict_class = np.argmax(result, axis = 1)
print(predict_class)