import abc
import gtimer as gt
import tensorflow as tf
import time

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


def add_to_summaries(value, name):
    value_summary = tf.Summary.Value(tag=name, simple_value=value)
    summaries.append(value_summary)
    
logdir = 'sac_' + time.strftime('%Y%m%d_%H%M') + '/'
summary_writer = None
summaries = []    
        
if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries',logdir)):
    os.mkdir(os.path.join('summaries',logdir))
summary_writer = tf.summary.FileWriter(os.path.join('summaries',logdir), tf.get_default_graph())

"""
filename_queue = tf.train.string_input_producer(["data.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1.0], [1.0]]
col1, col2 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10001):
        # Retrieve a single instance:
        example, label = sess.run([features, col2])
                #tensorboard
        add_to_summaries(example, "return_average")
        c = tf.Summary(value= summaries)
        summary_writer.add_summary(c, i)
        summaries = []

    coord.request_stop()
    coord.join(threads)
"""
with open('progress.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:
        #print(row['return-average'])
        if i == 4001:
            break
        
        add_to_summaries(float(row['return-average']), "return_average")
        add_to_summaries(float(row['qf-std']), "qf-std")
        add_to_summaries(float(row['qf-avg']), "qf-avg")
        add_to_summaries(float(row['vf-std']), "vf-std")
        add_to_summaries(float(row['vf-avg']), "vf-avg")
        epoch = row['epoch']
        c = tf.Summary(value= summaries)
        summary_writer.add_summary(c, int(epoch) + 6000)
        summaries = []
        
        i+=1
