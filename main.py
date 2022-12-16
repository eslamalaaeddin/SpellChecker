import re
from collections import Counter
import numpy as np
import pandas as pd


def process_data(file_name):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        words: a list containing all the words in the corpus (text file you read) in lower case.
    """
    words = []  # return this variable correctly

    ### START CODE HERE ###
    with open(file_name) as f:
        file_name_data = f.read()
    file_name_data = file_name_data.lower()
    words = re.findall('\w+', file_name_data)
    ### END CODE HERE ###

    return words


words = process_data('shakespeare.txt')
print('Count of words:' +  str(len(words)))
vocab = set(words)  # this will be your new vocabulary


# print(f"The first ten words in the text are: \n{word_l[0:10]}")
# print(f"There are {len(vocab)} unique words in the vocabulary.")


def get_count(word_l):
    '''
    Input:
        word_l: a set of words representing the corpus.
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    '''

    word_count_dict = {}  # fill this with word counts
    ### START CODE HERE
    word_count_dict = Counter(word_l)
    ### END CODE HERE ###
    return word_count_dict


word_count_dict = get_count(words)


# print(f"There are {len(word_count_dict)} key values pairs")
# print(f"The count for the word 'thee' is {word_count_dict.get('thee', 0)}")


def get_probs(word_count_dict):
    '''
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur.
    '''
    probs = {}  # return this variable correctly

    ### START CODE HERE ###
    m = sum(word_count_dict.values())
    for key in word_count_dict.keys():
        probs[key] = word_count_dict[key] / m
    ### END CODE HERE ###
    return probs


probs = get_probs(word_count_dict)


# print(f"Length of probs is {len(probs)}")
# print(f"P('thee') is {probs['thee']:.4f}")


def delete_letter(word, verbose=False):
    '''
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''

    delete_l = []
    split_l = []

    ### START CODE HERE ###
    for c in range(len(word)):
        split_l.append((word[:c], word[c:]))
    for a, b in split_l:
        delete_l.append(a + b[1:])
    ### END CODE HERE ###

    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l


# delete_word_l = delete_letter(word="cans", verbose=True)
# print(f"Number of outputs of delete_letter('at') is {len(delete_letter('at'))}")


def switch_letter(word, verbose=False):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    '''

    switch_l = []
    split_l = []

    ### START CODE HERE ###
    len_word = len(word)
    for c in range(len_word):
        split_l.append((word[:c], word[c:]))
    switch_l = [a + b[1] + b[0] + b[2:] for a, b in split_l if len(b) >= 2]
    ### END CODE HERE ###

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


# switch_word_l = switch_letter(word="eta",verbose=True)
# print(f"Number of outputs of switch_letter('at') is {len(switch_letter('at'))}")


def replace_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    '''

    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    split_l = []

    ### START CODE HERE ###
    for c in range(len(word)):
        split_l.append((word[0:c], word[c:]))
    replace_l = [a + l + (b[1:] if len(b) > 1 else '') for a, b in split_l if b for l in letters]
    replace_set = set(replace_l)
    replace_set.remove(word)
    ### END CODE HERE ###

    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l


# replace_l = replace_letter(word='can',verbose=True)
# print(f"Number of outputs of switch_letter('at') is {len(switch_letter('at'))}")


def insert_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []

    ### START CODE HERE ###
    for c in range(len(word) + 1):
        split_l.append((word[0:c], word[c:]))
    insert_l = [a + l + b for a, b in split_l for l in letters]
    ### END CODE HERE ###

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l


# insert_l = insert_letter('at', True)
# print(f"Number of strings output by insert_letter('at') is {len(insert_l)}")
# print(f"Number of outputs of insert_letter('at') is {len(insert_letter('at'))}")


def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    edit_one_set = set()

    ### START CODE HERE ###
    edit_one_set.update(delete_letter(word))
    if allow_switches:
        edit_one_set.update(switch_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    ### END CODE HERE ###

    return edit_one_set


# tmp_word = "at"
# tmp_edit_one_set = edit_one_letter(tmp_word)
# tmp_edit_one_l = sorted(list(tmp_edit_one_set))
# print(f"input word {tmp_word} \nedit_one_l \n{tmp_edit_one_l}\n")
# print(f"The type of the returned object should be a set {type(tmp_edit_one_set)}")
# print(f"Number of outputs from edit_one_letter('at') is {len(edit_one_letter('at'))}")


def edit_two_letters(word, allow_switches=True):
    '''
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''

    edit_two_set = set()

    ### START CODE HERE ###
    edit_one = edit_one_letter(word, allow_switches=allow_switches)
    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w, allow_switches=allow_switches)
            edit_two_set.update(edit_two)
    ### END CODE HERE ###

    return edit_two_set


# tmp_edit_two_set = edit_two_letters("a")
# tmp_edit_two_l = sorted(list(tmp_edit_two_set))
# print(f"Number of strings with edit distance of two: {len(tmp_edit_two_l)}")
# print(f"First 10 strings {tmp_edit_two_l[:10]}")
# print(f"Last 10 strings {tmp_edit_two_l[-10:]}")
# print(f"The data type of the returned object should be a set {type(tmp_edit_two_set)}")
# print(f"Number of strings that are 2 edit distances from 'at' is {len(edit_two_letters('at'))}")

# Python short circuits
# print([] and ["a", "b"])
# print([] or ["a", "b"])
#
# val1 = ["Most", "Likely"] or ["Less", "so"] or ["least", "of", "all"]  # selects first, does not evalute remainder
# print(val1)
# val2 = [] or [] or ["least", "of", "all"]  # continues evaluation until there is a non-empty list
# print(val2)


def get_corrections(word, probs, vocab, suggestions_count, verbose=False):
    '''
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''

    suggestions = []
    n_best = []

    ### START CODE HERE ###
    ## intersectoin means -> 'that are in the vocab'
    suggestions = list(
        (word in vocab and word) or edit_one_letter(word).intersection(vocab) or edit_two_letters(word).intersection(
            vocab))
    n_best = [(s, probs[s]) for s in list(reversed(suggestions))]

    n_best = sorted(
        n_best,
        key=lambda tpl: tpl[1],
        reverse=True)
    ### END CODE HERE ###

    # if verbose: print("suggestions (unsorted) = ", suggestions)

    return n_best[:suggestions_count]


# my_word = 'car'
#
# tmp_corrections = get_corrections(my_word, probs, vocab, suggestions_count=5, verbose=True)
# for i, word_prob in enumerate(tmp_corrections):
#     print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")


def correct(text):
    words = text.split(' ')
    words = [word.lower() for word in words]
    suggestion = ""
    for word in words:
        corrections = get_corrections(word, probs, vocab, suggestions_count=5, verbose=True)
        suggestion += corrections[0][0] + " "
        for i, word_prob in enumerate(corrections):
            print(f"{word_prob[0]} ==> probability: {word_prob[1]:.6f}")
        print('------------------------------------------')
    print(suggestion)

correct("I wakt tp gw tp schopl becoyse I wamt tb leqrn")


def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    '''
    Input:
        source: a string corresponding to the string you are starting with
        target: a string corresponding to the string you want to end with
        ins_cost: an integer setting the insert cost
        del_cost: an integer setting the delete cost
        rep_cost: an integer setting the replace cost
    Output:
        D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        med: the minimum edit distance (med) required to convert the source string to the target
    '''
    # use deletion and insert cost as  1
    m = len(source)
    n = len(target)
    # initialize cost matrix with zeros and dimensions (m+1,n+1)
    D = np.zeros((m + 1, n + 1), dtype=int)

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # Fill in column 0, from row 1 to row m, both inclusive
    for row in range(1, m + 1):  # Replace None with the proper range
        D[row, 0] = D[row - 1, 0] + del_cost

    # Fill in row 0, for all columns from 1 to n, both inclusive
    for col in range(1, n + 1):  # Replace None with the proper range
        D[0, col] = D[0, col - 1] + ins_cost

    # Loop through row 1 to row m, both inclusive
    for row in range(1, m + 1):

        # Loop through column 1 to column n, both inclusive
        for col in range(1, n + 1):

            # Intialize r_cost to the 'replace' cost that is passed into this function
            r_cost = rep_cost

            # Check to see if source character at the previous row
            # matches the target character at the previous column,
            if source[row - 1] == target[col - 1]:
                # Update the replacement cost to 0 if source and target are the same
                r_cost = 0

            # Update the cost at row, col based on previous entries in the cost matrix
            # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
            D[row, col] = min([D[row - 1, col] + del_cost, D[row, col - 1] + ins_cost, D[row - 1, col - 1] + r_cost])

    # Set the minimum edit distance with the cost found at row m, column n
    med = D[m, n]

    ### END CODE HERE ###
    return D, med


def test_min_edit_distance():
    source = 'play'
    target = 'stay'
    matrix, min_edits = min_edit_distance(source, target)
    print("minimum edits: ", min_edits, "\n")
    idx = list('#' + source)
    cols = list('#' + target)
    df = pd.DataFrame(matrix, index=idx, columns=cols)
    print(df)

    source = 'on'
    target = 'of'
    matrix, min_edits = min_edit_distance(source, target)
    print("minimum edits: ", min_edits, "\n")
    idx = list(source)
    idx.insert(0, '#')
    cols = list(target)
    cols.insert(0, '#')
    df = pd.DataFrame(matrix, index=idx, columns=cols)
    print(df)

    source = "eer"
    targets = edit_one_letter(source,
                              allow_switches=False)  # disable switches since min_edit_distance does not include them
    for t in targets:
        _, min_edits = min_edit_distance(source, t, 1, 1, 1)  # set ins, del, sub costs all to one
        if min_edits != 1: print(source, t, min_edits)

    source = "eer"
    targets = edit_two_letters(source,
                               allow_switches=False)  # disable switches since min_edit_distance does not include them
    for t in targets:
        _, min_edits = min_edit_distance(source, t, 1, 1, 1)  # set ins, del, sub costs all to one
        if min_edits != 2 and min_edits != 1: print(source, t, min_edits)
