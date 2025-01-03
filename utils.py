from gensim.models import KeyedVectors
import pickle
import pandas as pd
from translator import *

def get_dict(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    my_file = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned
    for i in range(len(my_file)):
        # indexing into the rows.
        en = my_file.iloc[i, 0]
        fr = my_file.iloc[i, 1]
        etof[en] = fr

    return etof

def dataset_preprocess():
    en_embeddings = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary = True)
    fr_embeddings = KeyedVectors.load_word2vec_format('data/wiki.fr.vec')
    
    
    # loading the english to french dictionaries
    en_fr_train = get_dict('data/en-fr.train.txt')
    en_fr_test = get_dict('data/en-fr.test.txt')
    
    english_set = set(en_embeddings.key_to_index)
    french_set = set(fr_embeddings.key_to_index)
    
    en_embeddings_subset = {}
    fr_embeddings_subset = {}
    french_words = set(en_fr_train.values())
    
    for en_word in en_fr_train.keys():
        fr_word = en_fr_train[en_word]
        if fr_word in french_set and en_word in english_set:
            en_embeddings_subset[en_word] = en_embeddings[en_word]
            fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]
    
    
    for en_word in en_fr_test.keys():
        fr_word = en_fr_test[en_word]
        if fr_word in french_set and en_word in english_set:
            en_embeddings_subset[en_word] = en_embeddings[en_word]
            fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]
    
    
    pickle.dump( en_embeddings_subset, open( "en_embeddings.p", "wb" ) )
    pickle.dump( fr_embeddings_subset, open( "fr_embeddings.p", "wb" ) )

def test_vocabulary(X, Y, R):
    '''
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    '''

    # The prediction is X times R
    pred = np.dot(X,R)

    # initialize the number correct to zero
    num_correct = 0

    # loop through each row in pred (each transformed embedding)
    for i in range(len(pred)):
        # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        pred_idx = nearest_neighbor(pred[i],Y)

        # if the index of the nearest neighbor equals the row of i... \
        if pred_idx == i:
            # increment the number correct by 1.
            num_correct += 1

    # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
    accuracy = num_correct / len(pred)


    return accuracy

def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    cos = -10
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)

    return cos



