from __future__ import print_function

import argparse
import os
import random

import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, Adagrad
from tqdm import tqdm

from model import DIIN
from optimizers.l2optimizer import L2Optimizer
from util import ChunkDataManager

from keras.utils import to_categorical
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Gym(object):
    def __init__(self,
                 model,
                 train_data,
                 test_data,
                 dev_data,
                 optimizers,
                 logger,
                 models_save_dir):

        self.model = model
        self.logger = logger

        ''' Data '''
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data
        self.model_save_dir = models_save_dir
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        ''' Optimizers '''
        self.optimizers = optimizers
        self.optimizer_id = -1
        self.current_optimizer = None
        self.current_switch_step = -1

    def switch_optimizer(self):
        self.optimizer_id += 1
        if self.optimizer_id >= len(self.optimizers):
            print('Finished training...')
            exit(0)

        self.current_optimizer, self.current_switch_step = self.optimizers[self.optimizer_id]
        self.model.compile(optimizer=self.current_optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.logger.set_model(self.model)
        print('Switching to number {} optimizer'.format(self.current_optimizer))

    def train(self, batch_size=70, eval_interval=500, shuffle=True):
        print('train:\t', [d.shape for d in self.train_data])
        print('test:\t',  [d.shape for d in self.test_data])
        print('dev:\t',   [d.shape for d in self.dev_data])

        # Initialize optimizer
        self.switch_optimizer()
        self.model.summary()

        # Start training
        train_step, eval_step, no_progress_steps = 0, 0, 0
        train_batch_start = 0
        best_loss = 1000.

        while True:
            if shuffle:
                random.shuffle(list(zip(train_data)))
            train_inputs = train_data[:-1]
            train_labels = train_data[-1]

            # Evaluate
            test_loss, dev_loss = self.evaluate(eval_step=eval_step, batch_size=batch_size)
            eval_step += 1

            # Switch optimizer if it's necessary
            no_progress_steps += 1
            if dev_loss < best_loss:
                best_loss = dev_loss
                no_progress_steps = 0

            if no_progress_steps >= self.current_switch_step:
                self.switch_optimizer()
                no_progress_steps = 0

            # Train eval_interval times
            for _ in tqdm(range(eval_interval)):
                [loss, acc] = model.train_on_batch(
                    [train_input[train_batch_start: train_batch_start + batch_size] for train_input in train_inputs],
                    train_labels[train_batch_start: train_batch_start + batch_size])
                self.logger.on_epoch_end(epoch=train_step, logs={'train_acc': acc, 'train_loss': loss})
                train_step += 1
                train_batch_start += batch_size
                if train_batch_start > len(train_inputs[0]):
                    train_batch_start = 0
                    # Shuffle the data after the epoch ends
                    if shuffle:
                        random.shuffle(list(zip(train_data)))

    def evaluate(self, eval_step, batch_size=None):
        [test_loss, test_acc] = model.evaluate(self.test_data[:-1], self.test_data[-1], batch_size=batch_size)
        [dev_loss,  dev_acc]  = model.evaluate(self.dev_data[:-1],  self.dev_data[-1],  batch_size=batch_size)
        self.logger.on_epoch_end(epoch=eval_step, logs={'test_acc': test_acc, 'test_loss': test_loss})
        self.logger.on_epoch_end(epoch=eval_step, logs={'dev_acc':  dev_acc,  'dev_loss':  dev_loss})
        model.save(self.model_save_dir + 'epoch={}-tloss={}-tacc={}.model'.format(eval_step, test_loss, test_acc))

        return test_loss, dev_loss

    def predict_res(self, models_dir, batch_size, predict_data):
        def find_best_model(dirs):
            max_index = 0
            max_res = 0
            for i in range(len(dirs)):
                index = dirs[i].index('acc=')
                if float(dirs[i][index+4:-6]) > max_res:
                    max_res = float(dirs[i][index+4:-6])
                    max_index = i
            return dirs[max_index]
        def to_pd(res):
            output_res = []
            for i in range(res.shape[0]):
                max_res_i = max([res[i][j] for j in range(res.shape[1])])
                for j in range(res.shape[1]):
                    if max_res_i == res[i][j]:
                        output_res.append(j)
            output_res = np.array(output_res)

            return output_res

        dirs = os.listdir(models_dir)
        best_model_dir = models_dir + find_best_model(dirs)
        model = self.model
        model.load_weights(best_model_dir)
        res = model.predict(predict_data, batch_size=batch_size)
        res = to_pd(res)
        res_pd = pd.DataFrame(res)
        res_pd = res_pd.reset_index()
        res_pd.columns = ['test_id','result']
        res_pd.to_csv('predict.csv', index=None)
        return res

    # def predict_and_save(self, data, file_path):
    #     predict = model.preditct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',                  default=42,        help='Batch size',                              type=int)
    parser.add_argument('--eval_interval',               default=500,       help='Evaluation Interval (#batches)',          type=int)
    parser.add_argument('--char_conv_kernel',            default=5,         help='Size of char convolution kernel',         type=int)
    parser.add_argument('--dropout_initial_keep_rate',   default=1.,        help='Initial keep rate of decaying dropout',   type=float)
    parser.add_argument('--dropout_decay_rate',          default=0.977,     help='Decay rate of dropout',                   type=float)
    parser.add_argument('--dropout_decay_interval',      default=10000,     help='Dropout decay interval',                  type=int)
    parser.add_argument('--l2_full_step',                default=100000,    help='Number of steps for full L2 penalty',     type=float)
    parser.add_argument('--l2_full_ratio',               default=9e-5,      help='L2 full penalty',                         type=float)
    parser.add_argument('--l2_diference_penalty',        default=1e-3,      help='L2 penalty applied on weight difference', type=float)
    parser.add_argument('--first_scale_down_ratio',      default=0.3,       help='First scale down ratio (DenseNet)',       type=float)
    parser.add_argument('--transition_scale_down_ratio', default=0.5,       help='Transition scale down ratio (DenseNet)',  type=float)
    parser.add_argument('--growth_rate',                 default=20,        help='Growth rate (DenseNet)',                  type=int)
    parser.add_argument('--layers_per_dense_block',      default=8,         help='Layers in one Dense block (DenseNet)',    type=int)
    parser.add_argument('--dense_blocks',                default=3,         help='Number of Dense blocks (DenseNet)',       type=int)
    parser.add_argument('--labels',                      default=2,         help='Number of output labels',                 type=int)
    parser.add_argument('--load_dir',                    default='./data',    help='Directory of the data',   type=str)
    parser.add_argument('--models_dir',                  default='models/', help='Where to save models',    type=str)
    parser.add_argument('--logdir',                      default='logs',    help='Tensorboard logs dir',    type=str)
    parser.add_argument('--word_vec_path', default='data/word-vectors.npy', help='Save path word vectors',  type=str)
    parser.add_argument('--omit_word_vectors',           default=0, type=int)
    parser.add_argument('--omit_chars',                  default=0, type=int)
    parser.add_argument('--omit_exact_match',            default=0, type=int)
    parser.add_argument('--train_word_embeddings',       default=True)
    parser.add_argument('--is_train',                    default=1, type=int)
    args = parser.parse_args()

    ''' Prepare data '''
    word_embedding_weights = np.load(args.word_vec_path)
    all_data = ChunkDataManager(load_data_path=os.path.join(args.load_dir+'/train')).load()
    predict_data = ChunkDataManager(load_data_path=os.path.join(args.load_dir+'/dev')).load()
    is_train = args.is_train

    if args.omit_exact_match == 1:
        del all_data[4:6]
        del predict_data[4:6]
    if args.omit_chars == 1:
        del all_data[2:4]
        del predict_data[4:6]

    train_data = [[] for i in range(len(all_data))]
    test_data = [[] for i in range(len(all_data))]
    dev_data = [[] for i in range(len(all_data))]
    for i, all_data_i in enumerate(all_data):
        train_data[i] = all_data_i[:80000]
        test_data[i] = all_data_i[80000:90000]
        dev_data[i] = all_data_i[90000:]

    train_data[-1] = to_categorical(train_data[-1])
    test_data[-1] = to_categorical(test_data[-1])
    dev_data[-1] = to_categorical(dev_data[-1])

    ''' Getting dimensions of the input '''
    chars_per_word = train_data[3].shape[-1] if not args.omit_chars else 0
    
    ''' Prepare the model and optimizers '''
    adam = L2Optimizer(Adam(), args.l2_full_step, args.l2_full_ratio, args.l2_diference_penalty)
    adagrad = L2Optimizer(Adagrad(lr=3e-4), args.l2_full_step, args.l2_full_ratio, args.l2_diference_penalty)
    sgd = L2Optimizer(SGD(lr=1e-4), args.l2_full_step, args.l2_full_ratio, args.l2_diference_penalty)
    # adam2 = L2Optimizer(Adam(lr=0.0001), args.l2_full_step, args.l2_full_ratio, args.l2_diference_penalty)

    model = DIIN(p=train_data[0].shape[-1],  # or None
                 h=train_data[1].shape[-1],  # or None
                 include_word_vectors=not args.omit_word_vectors,
                 word_embedding_weights=word_embedding_weights,
                 train_word_embeddings=args.train_word_embeddings,
                 include_chars=1 - args.omit_chars,
                 chars_per_word=chars_per_word,
                 char_conv_kernel_size=args.char_conv_kernel,
                 include_exact_match=1 - args.omit_exact_match,
                 dropout_initial_keep_rate=args.dropout_initial_keep_rate,
                 dropout_decay_rate=args.dropout_decay_rate,
                 dropout_decay_interval=args.dropout_decay_interval,
                 first_scale_down_ratio=args.first_scale_down_ratio,
                 transition_scale_down_ratio=args.transition_scale_down_ratio,
                 growth_rate=args.growth_rate,
                 layers_per_dense_block=args.layers_per_dense_block,
                 nb_dense_blocks=args.dense_blocks,
                 nb_labels=args.labels)

    ''' Initialize Gym for training '''
    gym = Gym(model=model,
              train_data=train_data, test_data=test_data, dev_data=dev_data,
              optimizers=[(adam, 4), (adagrad, 6), (sgd, 12)],
              logger=TensorBoard(log_dir=args.logdir),
              models_save_dir=args.models_dir)
    if is_train:
        gym.train(batch_size=args.batch_size, eval_interval=args.eval_interval, shuffle=True)
    else:
        print('predict-------------------------')
        predict_res = gym.predict_res(args.models_dir, args.batch_size, predict_data)

