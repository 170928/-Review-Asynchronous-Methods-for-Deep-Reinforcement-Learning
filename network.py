import tensorflow as tf
import numpy as np
from args import args


class ACNet(object):
    def __init__(self, sess, scope, args, globalAC=None):

        self.sess = sess
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.A_BOUND = args.action_bound
        self.ENTROPY_BETA = args.entropy_beta

        if scope == 'global_network':

            with tf.variable_scope(scope):
                self.states = tf.placeholder(tf.float32, [None, self.state_dim], 'State')
                self.actor_params, self.critic_params = self._build_net(scope)[-2:]

        else:
            with tf.variable_scope(scope):

                self.states =  tf.placeholder(tf.float32, [None, self.state_dim], 'State')
                self.actions = tf.placeholder(tf.float32, [None, self.action_dim], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.actor_params, self.critic_params = self._build_net(scope)

                '''
                base 와 critic Q(s, a)의 action-value의 값을 빼준다. 
                TD_error라고 부릅니다. 
                v 값은 worker의 critic의 추정치 입니다. 
                '''
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                '''
                A3C 에서는 TD_error의 mean을 critic loss 로써 사용합니다.
                '''
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                '''
                action bound를 곱해주는 것으로 tanh 의 output인 -1~1 범위를 action bound 의 범위로 바꾸어 줍니다.
                '''
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * self.A_BOUND[1], sigma + 1e-4

                '''
                continuos action의 random을 위해서,
                출력 mu 를 중심으로 sigma 범위 내에서 normal distribution에 따른 결과를 
                action으로 사용합니다. 
                '''
                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.actions)
                    '''
                    log( action _ prob ) * td 를 actor의 parameter update를 위한 gradient로 사용합니다. 
                    '''
                    exp_v = log_prob * tf.stop_gradient(td)
                    '''
                    entropy regulizer를 더해 주는 것으로 최종 actor 학습을 위한 loss 값을 만듭니다. 
                    '''
                    entropy = normal_dist.entropy()
                    self.exp_v = self.ENTROPY_BETA * entropy + exp_v
                    '''
                    negative loss 를 minimize 하는 것이 policy gradient의 gradient ascend 와 같음으로
                    - 를 붙여줍니다. 
                    '''
                    self.a_loss = tf.reduce_mean(-self.exp_v)



                '''
                Action은 action bound 내에서 결정되어야 하므로
                다음과 같은 clip_by_value 함수를 통해서 맞추어 줍니다.
                '''
                with tf.name_scope('choose_a'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), self.A_BOUND[0], self.A_BOUND[1])

                '''
                actor loss 는 actor의 parameter들에 대해서 미분
                critic loss 는 critic의 parameter들에 대해서 미분 
                '''
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.actor_params)
                    self.c_grads = tf.gradients(self.c_loss, self.critic_params)

            with tf.name_scope('sync'):
                '''
                global netwrok의 parameter들을 
                worker의 parameter에 할당합니다. 
                '''
                with tf.name_scope('worker_sync'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params, globalAC.actor_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_params, globalAC.critic_params)]

                '''
                global_netowkr의 parameter들을 worker가 가진 gradients 정보를 활용해서
                global actor와 critic을 업데이트 합니다. 
                아래의 operation들을 호출하는 것으로
                global network가 업데이트 됩니다.
                '''
                with tf.name_scope('global_sync'):
                    self.update_a_op = args.OPT_A.apply_gradients(zip(self.a_grads, globalAC.actor_params))
                    self.update_c_op = args.OPT_C.apply_gradients(zip(self.c_grads, globalAC.critic_params))

    def _build_net(self, scope):

        w_init = tf.random_normal_initializer(0., .1)

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.states, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, self.action_dim, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, self.action_dim, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.states, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')


        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return mu, sigma, v, a_params, c_params

    '''
    global network를 worker가 가진 gradient를 사용하여 업데이트 합니다. 
    '''
    def update_global(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    '''
    worker network 의 paramter를 global network의 parameter로 업데이트 합니다.
    '''
    def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    '''
    state 입력을 받아서 action을 출력합니다.
    '''
    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.states: s})
