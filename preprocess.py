from __future__ import print_function

import argparse
import io
import json
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import pickle as pkl

import jieba.posseg as psg


def pad(x, maxlen):
    if len(x) <= maxlen:
        pad_width = ((0, maxlen - len(x)), (0, 0))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    res = x[:maxlen]
    return np.array(res, copy=False)


class Preprocessor(object):

    def __init__(self):
        self.word_to_id = {}
        self.char_to_id = {}
        self.vectors = []
        self.part_of_speech_to_id = {}
        self.unique_words = set()
        self.unique_parts_of_speech = set()


    def get_all_words_with_parts_of_speech(self, all_sentences):
        def get_pos_from_sentence(sentence):
            pos = psg.cut(sentence)
            pos_list = []
            for ele in pos:
                pos_list += [ele.flag]

            return pos_list

        all_words = []
        all_pos = []
        for line in tqdm(all_sentences):
            l = line.split('\t')
            all_words += list(l[0]) + list(l[1])
    
            pos_s1 = get_pos_from_sentence(l[0])
            pos_s2 = get_pos_from_sentence(l[1])
            all_pos += pos_s1 + pos_s2
        
        self.unique_words = set(all_words)
        self.unique_parts_of_speech = set(all_pos)

    def load_word_vectors(self, file_path):

        w2v_model = pkl.load(open(file_path, 'rb'), encoding='iso-8859-1')
        words = list(self.unique_words)
        w2v_keys = w2v_model.keys()

        present_words = []
        not_present_words = []

        for word in words:
            if word in w2v_keys:
                present_words.append(word)
            else:
                not_present_words.append(word)
                print(word)

        print('present_words_num:', len(present_words), 'not_present_words_num:', len(not_present_words))

        vectors = []

        for present_word in present_words:
            vector = w2v_model[present_word]
            vectors.append(vector)

        vectors = np.array(vectors, dtype='float32', copy=False)
        return present_words, not_present_words, vectors

    def get_not_present_word_vectors(self, not_present_words, word_vector_size):
        res_words = []
        res_vectors = []
        for word in not_present_words:
            vec = np.random.uniform(size=word_vector_size)
            vec /= np.linalg.norm(vec, ord=2)
            res_words.append(word)
            res_vectors.append(vec)
        return res_words, res_vectors

    def init_word_to_vectors(self, vectors_file_path, needed_words, normalize=False, max_loaded_word_vectors=None):

        needed_words = set(needed_words)
        present_words, not_present_words, self.vectors = self.load_word_vectors(file_path=vectors_file_path)

        word_vector_size = self.vectors.shape[-1]
        
        not_present_words, not_present_vectors = self.get_not_present_word_vectors(not_present_words, word_vector_size)

        words = present_words + not_present_words
        self.vectors = list(self.vectors) + not_present_vectors

        print('Initializing word mappings...')
        self.word_to_id  = {word: i   for i, word in enumerate(words)}
        self.vectors = np.array(self.vectors, copy=False)
        
        assert len(self.word_to_id) == len(self.vectors)
        print(len(self.word_to_id), 'words in total are now initialized!')

    def init_parts_of_speech(self, parts_of_speech):
        self.part_of_speech_to_id = {part: i for i, part in enumerate(parts_of_speech)}
        print('Parts of speech:', parts_of_speech)

    def save_word_vectors(self, file_path):
        np.save(file_path, self.vectors)

    def get_sentences(self, line):
        return line[0], line[1]

    def get_syntactical_one_hot(self, sentence):
        syntactical_one_hot = [0] * len(self.unique_parts_of_speech)
        pos = psg.cut(sentence)
        pos_id_list = []
        for ele in pos:
            pos_id_list.append(self.part_of_speech_to_id[ele.flag])

        for pos_id in pos_id_list:
            syntactical_one_hot[pos_id] = 1

        return np.array(syntactical_one_hot, copy=False)

    def parse_sentence(self, sentence, max_words, chars_per_word):

        def char_features(word_ids, chars_per_word):
            chars = []
            for i in range(len(word_ids)):
                pre_ids = []
                post_ids = []
                if (i < chars_per_word):
                    pre_ids = [0] * (chars_per_word - i) + word_ids[:i]
                else:
                    pre_ids = word_ids[i-chars_per_word:i]
                if (i > len(word_ids) - chars_per_word - 1):
                    post_ids = word_ids[i:] + [0] * (i - len(word_ids) + chars_per_word)
                else:
                    post_ids = word_ids[i:i+chars_per_word]
                per_ids = pre_ids + post_ids
                chars.append(per_ids)
            return chars

        # Words
        words = list(sentence)
        word_ids = [self.word_to_id[word] for word in words]
        # Chars
        chars = char_features(word_ids, chars_per_word)
        
        return words, np.array(word_ids, copy=False), pad(chars, max_words)

    def parse_one(self, premise, hypothesis, max_words_p, max_words_h, chars_per_word):

        premise_words, premise_word_ids, premise_chars = self.parse_sentence(premise, max_words_p, chars_per_word)
        hypothesis_words, hypothesis_word_ids, hypothesis_chars = self.parse_sentence(hypothesis, max_words_h, chars_per_word)

        def calculate_exact_match(source_words, target_words):
            target_words = set(target_words)

            res = [0] * len(source_words)
            
            for i in range(len(source_words)):
                if source_words[i] in target_words:
                    res[i] = i
                    
            return np.array(res, copy=False)

        premise_exact_match    = calculate_exact_match(premise_words, hypothesis_words)
        hypothesis_exact_match = calculate_exact_match(hypothesis_words, premise_words)

        return (premise_word_ids, hypothesis_word_ids,
                premise_chars, hypothesis_chars,
                premise_exact_match, hypothesis_exact_match)


    def parse(self, data, max_words_p=32, max_words_h=32, is_train=True, chars_per_word=3):
        if is_train:
            res = [[], [], [], [], [], [], []]
        else:
            res = [[], [], [], [], [], []]

        for sample in tqdm(data):
            sample = sample.split('\t')

            premise, hypothesis = self.get_sentences(sample)
            sample_inputs = self.parse_one(premise, hypothesis, max_words_p, max_words_h, chars_per_word)

            if is_train:
                label = sample[2]
                sample_result = list(sample_inputs) + [label]
            else:
                sample_result = sample_inputs

            for res_item, parsed_item in zip(res, sample_result):
                res_item.append(parsed_item)

        res[0] = pad_sequences(res[0], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[1] = pad_sequences(res[1], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h
        res[4] = pad_sequences(res[4], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        res[5] = pad_sequences(res[5], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h
        
        return res

def preprocess(p, h, chars_per_word, preprocessor, save_dir,
               word_vector_save_path, normalize_word_vectors, max_loaded_word_vectors=None, word_vectors_load_path=None,
               train_path='./data/train.txt',
               dev_path='./data/dev.txt',
               save_path = './data/'):

    def load_data(file_path):
        f = open(file_path)
        data = f.readlines()
        f.close()
        return data

    train_data = load_data(train_path)
    dev_data = load_data(dev_path)

    all_sentences = train_data + dev_data


    preprocessor.get_all_words_with_parts_of_speech(all_sentences)
    print('Found', len(preprocessor.unique_words), 'unique words')
    print('Found', len(preprocessor.unique_parts_of_speech), 'unique parts of speech')

    preprocessor.init_word_to_vectors(vectors_file_path=word_vectors_load_path,
                                      needed_words=preprocessor.unique_words,
                                      normalize=normalize_word_vectors,
                                      max_loaded_word_vectors=max_loaded_word_vectors)

    preprocessor.init_parts_of_speech(parts_of_speech=preprocessor.unique_parts_of_speech)

    preprocessor.save_word_vectors(word_vector_save_path)

    data_train = preprocessor.parse(train_data, max_words_p=p, max_words_h=h, is_train=True, chars_per_word=chars_per_word)
    data_dev = preprocessor.parse(dev_data, max_words_p=p, max_words_h=h, is_train=False, chars_per_word=chars_per_word)


    def to_save(data, save_path, tag):
        print('Saving data of shapes:')
        for i, item in tqdm(enumerate(data)):
            np.save(save_path + tag + str(i) + '.npy', item)

    to_save(data_train, save_path+'train/', 'train')
    to_save(data_dev, save_path+'dev/', 'dev')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p',              default=60,         help='Maximum words in premise',            type=int)
    parser.add_argument('--h',              default=60,         help='Maximum words in hypothesis',         type=int)
    parser.add_argument('--chars_per_word', default=4,          help='Number of characters in one word',    type=int)
    parser.add_argument('--max_word_vecs',  default=None,       help='Maximum number of word vectors',      type=int)
    parser.add_argument('--save_dir',       default='data/',    help='Save directory of data',              type=str)
    parser.add_argument('--word_vec_load_path', default='./data/wx_vector_char.pkl',   help='Path to load word vectors',           type=str)
    parser.add_argument('--word_vec_save_path', default='data/word-vectors.npy', help='Path to save vectors', type=str)
    parser.add_argument('--normalize_word_vectors',      action='store_true')

    parser.add_argument('--train_path',         default='./data/train.txt')
    parser.add_argument('--dev_path',           default='./data/dev.txt')

    args = parser.parse_args()

    preprocessor = Preprocessor()


    preprocess(p=args.p, h=args.h, chars_per_word=args.chars_per_word,
                    preprocessor=preprocessor,
                    save_dir=args.save_dir,
                    word_vectors_load_path=args.word_vec_load_path,
                    normalize_word_vectors=args.normalize_word_vectors,
                    word_vector_save_path=args.word_vec_save_path,
                    max_loaded_word_vectors=args.max_word_vecs,
                    train_path='./data/train.txt',
                    dev_path='./data/dev.txt',
                    save_path = './data/')







