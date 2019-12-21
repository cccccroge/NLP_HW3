from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim.models.keyedvectors as word2vec


""" Define variables """
batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 128  # Number of samples to train on.

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

    return (input_texts, correct_texts)

train_input_texts, train_correct_texts = get_inputs_corrects(train_data_path, num_samples)
val_input_texts, val_correct_texts = get_inputs_corrects(val_data_path, int(num_samples / 4))


""" Process texts """
t = Tokenizer(filters='', split=' ')
t.fit_on_texts(train_input_texts + train_correct_texts + val_input_texts + val_correct_texts)
VOCAB_SIZE = len(t.index_word) + 1    # plus 0 for unknown words
index_word = t.index_word
word_index = t.word_index

train_input_sequences = t.texts_to_sequences(train_input_texts)
train_correct_sequences = t.texts_to_sequences(train_correct_texts)
val_input_sequences = t.texts_to_sequences(val_input_texts)
val_correct_sequences = t.texts_to_sequences(val_correct_texts)

train_input_sequences_pad = pad_sequences(train_input_sequences, maxlen=MAX_SEQ_LEN, padding='post')
train_correct_sequences_pad = pad_sequences(train_correct_sequences, maxlen=MAX_SEQ_LEN, padding='post')
val_input_sequences_pad = pad_sequences(val_input_sequences, maxlen=MAX_SEQ_LEN, padding='post')
val_correct_sequences_pad = pad_sequences(val_correct_sequences, maxlen=MAX_SEQ_LEN, padding='post')


""" Load Word2Vec model, create embedding matrix """
w2v_model = word2vec.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
WORD_VEC_DIM = w2v_model.vector_size
embedding_matrix = np.zeros((VOCAB_SIZE, WORD_VEC_DIM))
for word, index in t.word_index.items():
    if word in w2v_model.wv:
        vec = w2v_model.wv[word]
        embedding_matrix[index] = vec

print('Number of samples:', len(train_input_texts))
print('Number of train samples:', num_samples)
print('Number of vocab:', VOCAB_SIZE)
print('Max sequence length for inputs:', MAX_SEQ_LEN)


""" Prepare model data """
def get_model_data(input_seqs, correct_seqs):
    encoder_input_data = np.zeros(
        (len(input_seqs), MAX_SEQ_LEN),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_seqs), MAX_SEQ_LEN),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_seqs), MAX_SEQ_LEN, 1),
        dtype='float32')

    for i, (input_seq, correct_seq) in enumerate(zip(input_seqs, correct_seqs)):
        for t, word_idx in enumerate(input_seq):
            encoder_input_data[i, t] = word_idx

        for t, word_idx in enumerate(correct_seq):
            decoder_input_data[i, t] = word_idx
            if t > 0:
                decoder_target_data[i, t - 1, 0] = word_idx

    return (encoder_input_data, decoder_input_data, decoder_target_data)

train_encoder_input_data, train_decoder_input_data, train_decoder_target_data =\
    get_model_data(train_input_sequences_pad, train_correct_sequences_pad)
val_encoder_input_data, val_decoder_input_data, val_decoder_target_data =\
    get_model_data(val_input_sequences_pad, val_correct_sequences_pad)


""" Training model """
# encoder part
encoder_embedding_inputs = Input(shape=(None, ))
encoder_embedding_layer = Embedding(VOCAB_SIZE, WORD_VEC_DIM, weights=[embedding_matrix], trainable=False)
encoder_embedding_outputs = encoder_embedding_layer(encoder_embedding_inputs)

encoder_inputs = encoder_embedding_outputs
encoder = LSTM(latent_dim, return_sequences=False, return_state=True)
encoder_output, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder part
decoder_embedding_inputs = Input(shape=(None, ))
decoder_embedding_layer = Embedding(VOCAB_SIZE, WORD_VEC_DIM, weights=[embedding_matrix], trainable=False)
decoder_embedding_outputs = decoder_embedding_layer(decoder_embedding_inputs)

decoder_inputs = decoder_embedding_outputs
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs)

decoder_dense_layer = Dense(1, activation='softmax')
decoder_dense_outputs = decoder_dense_layer(decoder_outputs)

# train & save
train_model = Model([encoder_embedding_inputs, decoder_embedding_inputs], decoder_dense_outputs)
train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
train_model.summary()         
train_model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data))
train_model.save('../model/genious.h5')


""" Predict Model """
# encoder part
encoder_model = Model(encoder_embedding_inputs, encoder_states)

# decoder part
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states_outputs = [state_h, state_c]
decoder_dense_outputs = decoder_dense_layer(decoder_outputs)
decoder_model = Model(
    [decoder_embedding_inputs] + decoder_states_inputs,
    [decoder_dense_outputs] + decoder_states_outputs)

# predict sequences
def decode_sequence(input_seq):
    states_val = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_index['\t']

    stop_condition = False
    gen_len = 0
    output_text = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_val)

        sampled_token_index = int(output_tokens[0, -1, 0])
        sampled_word = index_word[sampled_token_index]
        output_text += (sampled_word + ' ')
        gen_len += 1

        # Exit condition
        if (sampled_word == '\n' or gen_len >= MAX_SEQ_LEN):
            stop_condition = True

        # Update the target sequence and states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_val = [h, c]

    return output_text

for text in val_input_texts[:50]:
    print('-')
    print('Input sentence:', text)

    seqs = t.texts_to_sequences([text])
    seqs = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding='post')
    seqs = seqs.astype('float32')
    seq = seqs[0]

    decoded_sentence = decode_sequence(seq)
    print('Decoded sentence:', decoded_sentence)
