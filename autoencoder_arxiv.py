# -*- coding: utf-8 -*-
"""
Yizhe Zhang

TextCNN
"""
## 152.3.214.203/6006

import os
GPUID = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
import sys
sys.path.append('/home/lqchen/work/textGAN/textGAN_public')

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
#from tensorflow.contrib import metrics
#from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
#from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb

from model import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx, _clip_gradients_seperate_norm
from denoise import *

profile = False
#import tempfile
#from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
#tf.logging.verbosity(1)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')


class Options(object):
    def __init__(self):
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn' #'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 41
        self.n_words = 27004
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 2
        self.embed_size = 300
        self.latent_size = 128
        self.lr = 1e-4
        
        self.rnn_share_emb = True
        self.additive_noise_lambda = 0.0
        self.bp_truncation = None
        self.n_hid = 100

        self.layer = 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 64
        self.max_epochs = 100
        self.n_gan = 128  # self.filter_size * 3
        self.L = 100

        self.optimizer = 'Adam' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None #None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.relu_w = False

        self.save_path = "./save_arxiv/" + "arxiv_" + str(self.n_gan) + "_dim_" + self.model + "_" + self.substitution + str(self.permutation)
        self.log_path = "./log_arxiv"
        self.print_freq = 1000
        self.valid_freq = 1000

        # batch norm & dropout
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1

        self.discrimination = False
        self.H_dis = 300
        self.ef_dim = 128

        self.sent_len = self.maxlen + 2*(self.filter_shape-1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape)/self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape)/self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        self.sentence = self.maxlen - 1
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def auto_encoder(x, x_org, is_train, opt, opt_t=None):
    # print x.get_shape()  # batch L
    with tf.variable_scope("pretrain"):    
    
        if not opt_t: opt_t = opt
        x_emb, W_norm = embedding(x, opt)  # batch L emb
	
        x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
        
        res = {}
        
        H, res = conv_encoder(x_emb, is_train, opt, res)
        
        H_mean, H_log_sigma_sq = vae_classifier_2layer(H, opt)
        eps = tf.random_normal([opt.batch_size, opt.ef_dim], 0, 1, dtype=tf.float32)
        H_dec = tf.add(H_mean, tf.multiply(tf.sqrt(tf.exp(H_log_sigma_sq)), eps))
        
        H_dec2 = tf.identity(H_dec)
         
        # print x_rec.get_shape()
        if opt.model == 'rnn_rnn':
            loss, rec_sent_1, _ = seq2seq(x, x_org, opt)
            _, rec_sent_2, _ = seq2seq(x, x_org, opt, feed_previous=True, is_reuse=True)
            #res['logits'] = logits
            res['rec_sents_feed_y'] = rec_sent_1
            res['rec_sents'] = rec_sent_2
            
            
        elif opt.model == 'cnn_rnn':
            # lstm decoder
            if opt.rnn_share_emb:
                loss, rec_sent_1, _ = lstm_decoder_embedding(H_dec2, x_org, W_norm, opt_t)
                # random_z = tf.random_normal([opt.batch_size, opt.ef_dim])  
                _, rec_sent_2, _ = lstm_decoder_embedding(H_dec2, x_org, W_norm, opt_t, feed_previous=True, is_reuse=True)
                
            else:
                loss, rec_sent_1, _ = lstm_decoder(H_dec2, x_org, opt_t)  #
                _, rec_sent_2, _ = lstm_decoder(H_dec2, x_org, opt_t, feed_previous=True, is_reuse=True)
                
            
            kl_loss= tf.reduce_mean(-0.5 * tf.reduce_mean(1 + H_log_sigma_sq 
                                                          - tf.square(H_mean) 
                                                          - tf.exp(H_log_sigma_sq), 1))
            loss += kl_loss
                                                     
            res['rec_sents_feed_y'] = rec_sent_1
            res['rec_sents'] = rec_sent_2
            
    
        else:
    
            # deconv decoder
            H_dec2 = tf.expand_dims(tf.expand_dims(H_dec, 1),1)
            loss, res = deconv_decoder(H_dec2, x_org, W_norm, is_train, opt_t, res)
            res['rec_sents'] = res['rec_sents'][:,(opt.filter_shape - 1): (opt.filter_shape - 1 + opt.sentence)]
            
    
    
    # *tf.cast(tf.not_equal(x_temp,0), tf.float32)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('kl_loss', kl_loss)
    summaries = [
                "learning_rate",
                "loss",
                # "gradients",
                # "gradient_norm",
                ]
    global_step = tf.Variable(0, trainable=False)


    train_op = layers.optimize_loss(
        loss,
        global_step = global_step,
        #aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        #framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate_decay_fn=lambda lr,g: tf.train.exponential_decay(learning_rate=lr, global_step = g, decay_rate=opt.decay_rate, decay_steps=3000),
        learning_rate=opt.lr,
        summaries = summaries
        )

    # optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)  # Or another optimization algorithm.
    # train_op = optimizer.minimize(
    #     loss,
    #     aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)


    return res, loss, train_op #, fake_gen


def run_model(opt, train, val, ixtoword):


    try:
        params = np.load('./param_g.npz')
        if params['Wemb'].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            opt.W_emb = params['Wemb']
        else:
            print('Emb Dimension mismatch: param_g.npz:'+ str(params['Wemb'].shape) + ' opt: ' + str((opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_org_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        is_train_ = tf.placeholder(tf.bool, name='is_train_')
        res_, loss_, train_op = auto_encoder(x_, x_org_, is_train_, opt)
        merged = tf.summary.merge_all()
        # opt.is_train = False
        # res_val_, loss_val_, _ = auto_encoder(x_, x_org_, opt)
        # merged_val = tf.summary.merge_all()

    #tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006
    #writer = tf.train.SummaryWriter(opt.log_path, graph=tf.get_default_graph())


    uidx = 0
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=tf.GraphOptions(build_cost_model=1))
    #config = tf.ConfigProto(device_count={'GPU':0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()



    run_metadata = tf.RunMetadata()


    with tf.Session(config = config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                #print([var.name[:-2] for var in t_vars])
                loader = restore_from_save(t_vars, sess, opt)
                print('\nLoad pretrain successfully\n')

            except Exception as e:
                print(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())
	
                
        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            # if epoch >= 10:
            #     print("Relax embedding ")
            #     opt.fix_emb = False
            #     opt.batch_size = 2
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                sents = [train[t] for t in train_index]

                sents_permutated = add_noise(sents, opt)

                #sents[0] = np.random.permutation(sents[0])
                
                if opt.model != 'rnn_rnn' and opt.model != 'cnn_rnn':
                    x_batch_org = prepare_data_for_cnn(sents, opt) # Batch L
                else:
                    x_batch_org = prepare_data_for_rnn(sents, opt) # Batch L

                if opt.model != 'rnn_rnn':
                    x_batch = prepare_data_for_cnn(sents_permutated, opt) # Batch L
                else:
                    x_batch = prepare_data_for_rnn(sents_permutated, opt, is_add_GO = False) # Batch L
                
                
                if profile:
                    _, loss = sess.run([train_op, loss_], feed_dict={x_: x_batch, x_org_: x_batch_org, is_train_:1},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                else:
                    _, loss = sess.run([train_op, loss_], feed_dict={x_: x_batch, x_org_: x_batch_org, is_train_:1})

                

                if uidx % opt.valid_freq == 1:
                    is_train = None
                    valid_index = np.random.choice(len(val), opt.batch_size)
                    val_sents = [val[t] for t in valid_index]

                    val_sents_permutated = add_noise(val_sents, opt)


                    if opt.model != 'rnn_rnn' and opt.model != 'cnn_rnn':
                        x_val_batch_org = prepare_data_for_cnn(val_sents, opt)
                    else:
                        x_val_batch_org = prepare_data_for_rnn(val_sents, opt)

                    if opt.model != 'rnn_rnn':
                        x_val_batch = prepare_data_for_cnn(val_sents_permutated, opt)
                    else:
                        x_val_batch = prepare_data_for_rnn(val_sents_permutated, opt, is_add_GO=False)

                    loss_val = sess.run(loss_, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org, is_train_:is_train })
                    print("Validation loss %f " % (loss_val))
                    res = sess.run(res_, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org, is_train_:is_train })
                    np.savetxt('./text_arxiv/rec_val_words.txt', res['rec_sents'], fmt='%i', delimiter=' ')
                    print "Sent:" + u' '.join([ixtoword[x] for x in res['rec_sents'][0] if x != 0]).encode('utf-8').strip()
                    if opt.discrimination:
                        print ("Real Prob %f Fake Prob %f" % (res['prob_r'], res['prob_f']))
                        
                    summary = sess.run(merged, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org, is_train_:is_train })
                    test_writer.add_summary(summary, uidx)
                    is_train = True
                    
                
                if uidx % opt.print_freq == 1:
                    #pdb.set_trace()
                    print("Iteration %d: loss %f " % (uidx, loss))
                    res = sess.run(res_, feed_dict={x_: x_batch, x_org_: x_batch_org, is_train_:1})
                    np.savetxt('./text_arxiv/rec_train_words.txt', res['rec_sents'], fmt='%i', delimiter=' ')
                    print "Sent:" + u' '.join([ixtoword[x] for x in res['rec_sents'][0] if x != 0]).encode('utf-8').strip()
                    summary = sess.run(merged, feed_dict={x_: x_batch, x_org_: x_batch_org, is_train_:1})
                    train_writer.add_summary(summary, uidx)
                    # print res['x_rec'][0][0]
                    # print res['x_emb'][0][0]
                    if profile:
                        tf.contrib.tfprof.model_analyzer.print_model_analysis(
                        tf.get_default_graph(),
                        run_meta=run_metadata,
                        tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

            saver.save(sess, opt.save_path, global_step=epoch)



def main():
    #global n_words
    # Prepare training and testing data
    #loadpath = "./data/three_corpus_small.p"
    loadpath = "./data/arxiv.p"
    x = cPickle.load(open(loadpath, 'rb'))
    train, val, test = x[0], x[1], x[2]
    train_text, val_text, test_text = x[3], x[4], x[5]
    train_lab, val_lab, test_lab = x[6], x[7], x[8]
    wordtoix, ixtoword = x[9], x[10]
    # ixtoword, _ = cPickle.load(open('vocab_cotra.pkl','rb'))
    # ixtoword = {i:x for i,x in enumerate(ixtoword)}

    opt = Options()

    print dict(opt)
    opt.n_words = len(ixtoword) + 1
    ixtoword[opt.n_words -1] = 'GO_'
    print('Total words: %d' % opt.n_words)
    
    run_model(opt, train, val, ixtoword)

    # model_fn = auto_encoder
    # ae = learn.Estimator(model_fn=model_fn)
    # ae.fit(train, opt , steps=opt.max_epochs)


#
# def main(argv=None):
#     learn_runner.run(experiment_fn, FLAGS.train_dir)

if __name__ == '__main__':
    main()
