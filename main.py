
# Reference Codes
# 김태훈 님의 코드를 참조하였습니다.  https://github.com/carpedm20/a3c-tensorflow/
# https://github.com/hongzimao/a3c
# https://github.com/hiwonjoon/tf-a3c-gpu
# https://github.com/carpedm20/a3c-tensorflow
# https://github.com/stefanbo92/A3C-Continuous
# 감사합니다.



import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Argument Setting")
parser.add_argument("--LR_actor", default = 0.0001)
parser.add_argument("--LR_critic", default = 0.001)
parser.add_argument("--GAMMA", default = 0.99)
parser.add_argument("--GAME", default = "Pendulum-v0")
parser.add_argument("--MAX_EP_STEP", default = 20000)
parser.add_argument("--MAX_GLOBAL_STEP", default = 20000)
parser.add_argument("--UPDATE_GLOBAL_ITER", default = 10)
parser.add_argument("--ENTROPY_BETA", default = 0.01)
parser.add_argument("--N_WORKERS", default = 10)
args = parser.parse_args()

OUTPUT_GRAPH  = True
LOG_DIR = './log'


env = gym.make(args.GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(args.LR_actor, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(args.LR_critic, name='RMSPropC')
        GLOBAL_AC = ACNet('global_network')  # we only need its params
        workers = []
        # Create worker
        for i in range(args.N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

