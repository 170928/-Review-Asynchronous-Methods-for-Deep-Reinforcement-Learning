from network import ACNet
import gym
import numpy as np


class Worker(object):

    '''
    class ACNet(object):
        def __init__(self, sess, scope, args, globalAC=None):
    '''

    def __init__(self, name, globalAC, args, sess):

        self.env = gym.make(args.GAME).unwrapped
        self.MAX_GLOBAL_EP = args.MAX_GLOBAL_EP
        self.MAX_EP_STEP = args.MAX_EP_STEP
        self.GAMMA = args.GAMMA
        self.UPDATE_GLOBAL_ITER = args.UPDATE_GLOBAL_ITER
        self.name = name
        self.sess = sess

        self.AC = ACNet(sess, name, args, globalAC)


    def setCoord(self, coord):
        self.COORD = coord

    def work(self, GLOBAL_RUNNING_R, GLOBAL_EP):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        '''
        tf.train.Coordinator() 를 사용하여, 쓰레드를 생성 하였으므로,
        해당 쓰레드가 적절히 종료 될 수 있도록
        루프문에 should_stop() 의 반환이 True일때 
        종료되는 루프를 기반으로 쓰레드를 만듭니다.
        '''
        while not self.COORD.should_stop() and GLOBAL_EP < self.MAX_GLOBAL_EP:

            s = self.env.reset()
            ep_r = 0
            for ep_t in range(self.MAX_EP_STEP):

                if self.name == 'W_0':
                    self.env.render()

                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == self.MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize

                if total_step % self.UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.states: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.states: buffer_s,
                        self.AC.actions: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break