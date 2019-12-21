from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import Word2Vec


""" Define variables """
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

MAX_SEQ_LEN = 50
VOCAB_SIZE = 0  # obtain after tokenizer is fit
WORD_VEC_DIM = 0 # obtain after w2v model is loaded

# Path to the data txt file on disk.
train_data_path = 'train.txt'
val_data_path = 'validation.txt'
word2vec_path = '../model/GoogleNews-vectors-negative300.bin'


""" Process files """
def get_inputs_corrects(data_path, num_samples):
    input_texts = []
    correct_texts = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[: min(num_samples, len(lines))]:  # need to -1?
        texts = line.split('\t')
        input_text = texts[0]
        correct_text = texts[1]
        correct_text = '\t ' + correct_text + ' \n'   # tab as <Start>, next line as <End>
        input_texts.append(input_text)
        correct_texts.append(correct_text)

train_input_texts, train_correct_texts = get_inputs_corrects(train_data_path, num_samples)
val_input_texts, val_correct_texts = get_inputs_corrects(val_data_path, num_samples / 4)


""" Process texts """
t = Tokenizer(filters='', split=' ')
t.fit_on_texts(train_input_texts + train_correct_texts + val_input_texts + val_correct_texts)
VOCAB_SIZE = t.num_words + 1    # plus 0 for unknown words
index_word = t.index_word

train_input_sequences = t.texts_to_sequences(train_input_texts)
train_correct_sequences = t.texts_to_sequences(train_correct_texts)
val_input_sequences = t.texts_to_sequences(val_input_texts)
val_correct_sequences = t.texts_to_sequences(val_correct_texts)

train_input_sequences_pad = pad_sequences(train_input_sequences, maxlen=MAX_SEQ_LEN, padding='post')
train_correct_sequences_pad = pad_sequences(train_correct_sequences, maxlen=MAX_SEQ_LEN, padding='post')
val_input_sequences_pad = pad_sequences(val_input_sequences, maxlen=MAX_SEQ_LEN, padding='post')
val_correct_sequences_pad = pad_sequences(val_correct_sequences, maxlen=MAX_SEQ_LEN, padding='post')


""" Load Word2Vec model, create embedding matrix """
w2v_model = Word2Vec.load(word2vec_path)
WORD_VEC_DIM = w2v_model.vector_size
embedding_matrix = np.zeros((VOCAB_SIZE, WORD_VEC_DIM))
for word, index in t.word_index.items():
    vec = w2v_model.wv[word]
    if vec is not None:
        embedding_matrix[index] = vec

print('Number of samples:', len(train_input_texts))
print('Number of train samples:', num_samples)
print('Number of vocab:', VOCAB_SIZE - 1)
print('Max sequence length for inputs:', MAX_SEQ_LEN)


""" Prepare model data """
encoder_input_data = np.zeros(
    (len(train_input_sequences_pad), MAX_SEQ_LEN),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(train_input_sequences_pad), MAX_SEQ_LEN),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(train_input_sequences_pad), MAX_SEQ_LEN, VOCAB_SIZE),
    dtype='float32')

for i, (input_seq, correct_seq) in enumerate(zip(train_input_sequences_pad, train_correct_sequences_pad)):
    for t, word_idx in enumerate(input_seq):
        encoder_input_data[i, t] = word_idx

    for t, word_idx in enumerate(correct_seq):
        encoder_input_data[i, t] = word_idx
        if t > 0:
            decoder_target_data[i, t - 1, word_idx] = 1.


""" Create training model """
# encoder part
encoder_embedding_inputs = Input(shape=(None, MAX_SEQ_LEN))
encoder_embedding_layer = Embedding(VOCAB_SIZE, WORD_VEC_DIM, weights=[embedding_matrix], trainable=False)
encoder_embedding_outputs = encoder_embedding_layer(encoder_embedding_inputs)

encoder_inputs = encoder_embedding_outputs
encoder = LSTM(latent_dim, return_sequences=False, return_state=True)
encoder_output, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder part
decoder_embedding_inputs = Input(shape=(None, MAX_SEQ_LEN))
decoder_embedding_layer = Embedding(VOCAB_SIZE, WORD_VEC_DIM, weights=[embedding_matrix], trainable=False)
decoder_embedding_outputs = decoder_embedding_layer(decoder_embedding_inputs)

decoder_inputs = decoder_embedding_outputs
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs)

decoder_dense_layer = Dense(VOCAB_SIZE, activation='softmax')
decoder_dense_outputs = decoder_dense_layer(decoder_outputs)

# train & save
train_model = Model([encoder_embedding_inputs, decoder_embedding_inputs], decoder_dense_outputs)
train_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_model.fit()


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,    # WHY: how to use _ as tuple elements
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('../model/s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)