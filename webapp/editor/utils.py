import numpy as np
import json
from keras.utils import np_utils

with open("editor/saved_model/vocab_clenaed-50.json", encoding="utf-8") as f:
  json_dump = json.load(f)
  char_to_int = json_dump["char_to_int"]
  int_to_char = json_dump["int_to_char"]
  int_to_char = {int(k):int_to_char[k] for k in int_to_char}
  stop_chars_int = [1, 3, 4, 5, 6, 7, 18, 19, 47, 49]
  max_len = 50
  n_vocab = len(int_to_char)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # select a character based on probability
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_sample(model, model_desc, text=None, 
                    temperatures=[0.7, 0.7, 0.6, 0.6,0.5, 0.5, 0.4, 0.4]):
  
    samples = []
    original_text = text


    # run for each temperature value
    for diversity in temperatures:

        # if custom text is provided, use lower case chars
        # make that given text has character present in vocab (50 chars)
        # pad it with space, if custome text is shorter than max lenght
        text = original_text
        text = " "*max_len + text.lower()
        # slice last max len chars
        text = text[-max_len:]

        # set the seed
        seed = text
        # tokenize the text
        tokenized_text = [char_to_int[c] for c in text]

        # conver to write shape: (1, max_len, n_vocab)
        x_seq = np_utils.to_categorical(tokenized_text, num_classes=n_vocab)
        # add batch axis as it was missing
        x_seq = np.expand_dims(x_seq, axis = 0)

        # generate 200 chars
        for i in range(200):
            # get the predictions orignal shape of prediction is (1, max_vocab), take first test sample from batch axis. final shape (50, )
            preds = model.predict(x_seq, verbose=0)[0]

            # use temperature based sampling
            next_index = sample(preds, diversity)
            # untokenize the index to get the character
            next_char = int_to_char[next_index]
            # append the char
            text = text + next_char
            if next_index in stop_chars_int: break
            # prepare a new array for prediction. shape (1, 1, 1).
            new_arr = np_utils.to_categorical(next_index, num_classes=n_vocab).reshape(1, 1, -1)
            # append it to current X, and take all elements other than first. (sliding window approach)
            # use this for prediction next iteration
            x_seq = np.concatenate([x_seq, new_arr], axis = 1)[:, 1:, :]

        # append to array to return
        samples.append((model_desc, diversity, seed, text))

    # return the all the samples
    return samples