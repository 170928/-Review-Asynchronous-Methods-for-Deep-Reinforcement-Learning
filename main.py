
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
from network import ACNet
from trainWorker import Worker
from args import args as ag
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

GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

if __name__ == "__main__":
    sess = tf.Session()

    with tf.device("/cpu:0"):

        OPT_A = tf.train.RMSPropOptimizer(args.LR_actor, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(args.LR_critic, name='RMSPropC')

        argsSet = ag(N_S, N_A, A_BOUND, args.ENTROPY_BETA, OPT_A, OPT_C, args.GAME, args.MAX_EP_STEP, args.MAX_GLOBAL_STEP, args.GAMMA, args.UPDATE_GLOBAL_ITER)

        GLOBAL_AC = ACNet(sess, scope = 'global_network', args = argsSet)  # we only need its params

        #-----------------workers-----------------
        workers = []
        for i in range(args.N_WORKERS):
            i_name = 'W_%i' % i
            '''
                def __init__(self, name, globalAC, args, GLOBAL_RUNNING_R, GLOBAL_EP, sess):            '''

            workers.append(Worker(i_name, GLOBAL_AC, argsSet, sess))

    '''
    
    Tensorflow에서는 sess 내의 다수 쓰레드들이 동시에 멈춰질 수 있어야하며 예외처리들이 이루어 질 수 있도록 하는 함수를 지원한다.
    이 함수가 tf.train.Coordinator() 라는 함수이다.
    이 조정자를 활용하면 아래와 같은 핵심 함수들이 사용 가능하다.
    (1) should_stop() : 쓰레드들이 정지되어야 한다면 True 값을 반환한다 
    (2) request_stop() : 쓰레드들이 정지되어야 함을 요청한다
    (3) join() : 특정 쓰레드들이 멈출 때까지 기다린다.
    
    조정자의 사용 방법은 다음과 같다
      1. Coordinator 객체를 생성한다.
      2. coordinator를 사용하는 쓰레드들을 생성한다.
    이때, 일반적으로 thread 들은 should_stop이 True를 반환할 때 멈추는 루프를 구성한다.
    
    '''

    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    '''
    그래프를 저장 하는 부분
    '''

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)


    '''
    Worker에 대한 thread를 만드는 부분
    '''

    worker_threads = []

    for worker in workers:
        worker.setCoord(COORD)

    for worker in workers:
        # worker내 work() 함수는 thread를 시작하는 함수
        # coord.should_stop 으로 작동하는 함수를 job으로 지칭하고
        # threading.Thread의 target으로 넣어준다.
        job = lambda: worker.work(GLOBAL_RUNNING_R, GLOBAL_EP)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)

    '''
    Coord.join 함수를 통해서 worker_threads 라는 list내의 thread 들이 멈출떄까지
    모두 기다린 다음 plot을 통해서 결과르 출력한다.
    '''
    COORD.join(worker_threads)


#    plt.plot(np.arange(len(global_rewards)), global_rewards)
#    plt.xlabel('step')
#    plt.ylabel('Total moving reward')
#    plt.show()

