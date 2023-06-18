import pickle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


def save_model():
    word2vec_glove_file = get_tmpfile("D:/nlp-project/src/glove.6B.100d.txt")
    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
    filename = 'models/glove2word2vec_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    return model


def get_model():
    filename = 'models/glove2word2vec_model.sav'
    model = pickle.load(open(filename, 'rb'))
    return model
