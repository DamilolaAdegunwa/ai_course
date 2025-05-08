# Make sure to install the necessary packages first
# pip install --upgrade pip
# pip install tensorflow

from typing import List

from tf_keras.preprocessing.text import Tokenizer  # worked!
# from keras.src.legacy.preprocessing.text import Tokenizer  # worked!
# from tf_keras.src.preprocessing.text import Tokenizer  # worked!

sentence = ["John likes to watch movies. Mary likes movies too."]


def print_bow(statement: List[str]) -> None:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(statement)
    sequences = tokenizer.texts_to_sequences(statement)
    word_index = tokenizer.word_index
    bow = {}
    for key in word_index:
        bow[key] = sequences[0].count(word_index[key])

    print(f"Bag of word sentence 1:\n{bow}")
    print(f"We found {len(word_index)} unique tokens.")


print_bow(sentence)
