"""Classification system.

Correctly able to find all the words in a given high quality wordsearch puzzle image, however cannot find all the words in a low quality image.

version: v1.0
"""

from typing import List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from utils import utils
from utils.utils import Puzzle
import scipy.linalg

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)

#This function loads the data and eigenvectors, creating the eigenvectors if there are none already. It then performs PCA reduction using N_DIMENSIONS number of dimensions.
def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:

    v = np.array(model["eigenvectors"])
    if v.size == 0:
        v = find_evectors(data, N_DIMENSIONS)
        model["eigenvectors"] = v.tolist()

    pcatrain_data = np.dot((data - np.mean(data)), v)

    return pcatrain_data

#This function finds the eigenvectors of a given dataset and number of dimensions.
def find_evectors(data, pcaNum):
    covx = np.cov(data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - pcaNum, N - 1))
    v = np.fliplr(v)

    return v

#The dictionary "model" is created and the relevant data is added to it.
def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["eigenvectors"] = [()]
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model

#The data is classified square-by-square and the resulting labels are returned.
def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    train = np.array(model["fvectors_train"])
    train_labels = np.array(model["labels_train"])

    x = np.dot(fvectors_test, train.transpose())
    modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest = np.argmax(dist, axis=1)

    return train_labels[nearest]


#The words in the list of labels are found and the positions of each of these is returned.
def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:

    word_pos = []

    for word in words:
        word = toUpper(word)
        searchList = complete_search(labels, word)
        for search in searchList:
            word_pos.append(search)
    return word_pos


#For a given word, every label is checked to see if a word can be found at that position.
def complete_search(array, word):
    height = find_height(array)
    width = len(array[0])
    temp = []
    found = False
    estimate = False
    #First, direct matches are searched for - only words with the exact letters as in the labels are returned.
    for i in range(height):
        for j in range(width):
            found, temp = directional_search(array, word, i, j, height, width, estimate, temp, found)

    #If the word cannot be found by the method above, then estimations are made - if one letter is wrong it is assumed that this is still the word.
    #And if the word still cannot be found, then it is replaced by an 'empty' tuple of (0,0,0,0).
    if not found:
        estimate = True
        for i in range(height):
            for j in range(width):
                if not found:
                    found, temp = directional_search(array, word, i, j, height, width, estimate, temp, found)
        if not found:
            temp.append((0,0,0,0))
    return temp

#Every position around a given letter is checked for the word being searched for - all 8 directions.
#When the positions are checked, they are checkded to ensure they both do not spill over the edge of the puzzle, and that they contain the word.
#Then the words are added to the temp array, and found is updated to true. If they are not found, nothing happens.
def directional_search(array, word, i, j, height, width, estimate, temp, found):
    if checkBelow(array, word, i, j, height, width, estimate):
        temp.append((i, j, i+len(word)-1, j))
        found = True
    elif checkAbove(array, word, i, j, height, width, estimate):
        temp.append((i, j, i-len(word)+1, j))
        found = True
    elif checkRight(array, word, i, j, height, width, estimate):
        temp.append((i, j, i, j+len(word)-1))
        found = True
    elif checkLeft(array, word, i, j, height, width, estimate):
        temp.append((i, j, i, j-len(word)+1))
        found = True
    elif checkDiagBR(array, word, i, j, height, width, estimate):
        temp.append((i, j, i+len(word)-1, j+len(word)-1))
        found = True
    elif checkDiagBL(array, word, i, j, height, width, estimate):
        temp.append((i, j, i+len(word)-1, j-len(word)+1))
        found = True
    elif checkDiagTR(array, word, i, j, height, width, estimate):
        temp.append((i, j, i-len(word)+1, j+len(word)-1))
        found = True
    elif checkDiagTL(array, word, i, j, height, width, estimate):
        temp.append((i, j, i-len(word)+1, j-len(word)+1))
        found = True
    return found, temp

#The function to compare a string and an array of letters to see if they match exactly.
def compare(string, array):
    for i in range(len(string)):
        if string[i] != array[i]:
            return False
    return True

#Function that compare string and an array of letters to see if they match - allowing for a single letter error in the match-up.
def compareEstimate(string, array):
    incorrect = 0
    for i in range(len(string)):
        if string[i] != array[i]:
            incorrect += 1
    if incorrect > 1:
        return False
    else:
        return True


#The 'check' functions below all take position of the provided i,j point in the array and return True if both the word would fit in the puzzle,
#and if the word is a match in the direction searched.
#They return False in any other case.
def checkBelow(array, word, i, j, height, width, estimate):
    temp = []
    if i+len(word)-1 < height:
        for x in range(len(word)):
            temp.append(array[i+x][j])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False

def checkAbove(array, word, i, j, height, width, estimate):
    temp = []
    if i-len(word)+1 > -1:
        for x in range(len(word)):
            temp.append(array[i-x][j])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False

def checkRight(array, word, i, j, height, width, estimate):
    temp = []
    if j+len(word)-1 < width:
        for x in range(len(word)):
            temp.append(array[i][j+x])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False

def checkLeft(array, word, i, j, height, width, estimate):
    temp = []
    if j-len(word)+1 > -1:
        for x in range(len(word)):
            temp.append(array[i][j-x])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False

def checkDiagBR(array, word, i, j, height, width, estimate):
    temp = []
    if i+len(word)-1 < height and j+len(word)-1 < width:
        for x in range(len(word)):
            temp.append(array[i+x][j+x])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False

def checkDiagBL(array, word, i, j, height, width, estimate):
    temp = []
    if i+len(word)-1 < height and j-len(word)+1 > 0:
        for x in range(len(word)):
            temp.append(array[i+x][j-x])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False

def checkDiagTR(array, word, i, j, height, width, estimate):
    temp = []
    if i-len(word)+1 > 0 and j+len(word)-1 < width:
        for x in range(len(word)):
            temp.append(array[i-x][j+x])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False

def checkDiagTL(array, word, i, j, height, width, estimate):
    temp = []
    if i-len(word)+1 > 0 and j-len(word)+1 > 0:
        for x in range(len(word)):
            temp.append(array[i-x][j-x])
        if (compare(word, temp) and not estimate) or (compareEstimate(word, temp) and estimate):
            return True
    return False


#function to find height of 2d array
def find_height(array):
    count = 0
    for i in array:
        count += 1
    return count

#This function ensures that every word letter and label letter are capitalised, which eliminates errors with mismatching cases.
def toUpper(string):
    upper = ""
    for i in range(len(string)):
        if string[i] >= 'a' and string[i] <= 'z':
            upper += chr(ord(string[i]) - 32)
        else:
            upper += string[i]
    return upper