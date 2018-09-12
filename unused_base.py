import tensorflow as tf

try:
  import queue
except:
  from six.moves import queue

class BaseAgent(object):

    def __init__(self, model_fn, env, config):

        self.env = env
        self.config = config
        self.task = config.task
        self.worker_device = "/job:worker/task:{}/cpu:0".format(self.task)

        '''

        From :: tensorflowkorea.gitbooks.io

         일반적으로 ''데이터 병렬화(data parallelism)'' 로 명명되는 훈련 방식은
         worker직무의 여러개의 작업이 하나의 모델에 대하여, 데이터의 각기 다른 일부를 이용하여 ps 에서 생성된 공유변수를 병렬적으로 업데이트 시키는 방식이다.
         모든 작업들은 각각 다른 머신에서 동작한다.
         텐서플로우에서 이러한 훈련 방식을 구현하는 방법은 여러가지가 있는데, 텐서플로우는 복제된 모델을 간단하게 생성할 수 있도록 도와주는 라이브러리를 구축하였다.
         시도 가능한 방법은 아래와 같다:

         [그래프간 복제(Between-graph replication)] 
            이 방법에서는 각 /job:worker 마다 별도의 클라이언트가 존재. 
            각 클라이언트는 변수를 포함해서 그래프를 복제한다.  
            그래프의 복사본은 /job:worker 의 로컬 작업에서 사용된다.
        '''

        with tf.device( tf.train.replica_device_setter(1, worker_device= self.worker_device)):
            # global network를 생성합니다.
            with tf.variable_scope("global"):
                self.network = model_fn()
                self.global_step = tf.get_variable(
                    name = "global_step", shape = [], dtype= tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable =False)

            # worker networks 를 생성합니다.
            with tf.device(self.worker_device):
                with tf.variable_scope("local"):
                    self.local_network = model_fn()
                    self.local_network.global_step = self.global_step

                # env.action_space.n 이므로 discrete action space 의 gymAI environment에서의
                # action 을 고려합니다.
                self.actions = tf.placeholder(tf.float32, [None, env.action_space.n], name = "action")
                self.rewards = tf.placeholder(tf.float32, [None], name = "reward")


    def build_shared_grad(self):
        # loss fuction 을 local_network_var_list의 변수에 대해서 미분합니다.
        self.grads = tf.gradients(self.loss, self.local_network.var_list)

        # tf.clip_by_global_norm을 통해서 gradient를 자릅니다.
        # Clips values of multiple tensors by the ratio of the sum of their norms.
        clipped_grads, _ = tf.clip_by_global_norm(self.grads, self.config.max_grad_norm)

        # sefl.sync operation은 local_network ( 논문에 따르면 worker network )들의 parameter를
        # global netwrok의 paramter로 할당하는 operation을 의미합니다.
        self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.network.var_list)])

        grads_and_vars = list(zip(clipped_grads, self.network.var_list))
        inc_step = self.global_step.assign_add(tf.shape(self.local_network.x)[0])

        # AdamOptimizer를 통해서 학습하는 과정에서
        # 학습률을 동일하게 적용하는 것이 아니라
        # decay learning rate 를 적용한다.
        # global step 에따라서 start rate 부터 decay step 마다 일정하게 변화 시킵니다.
        self.lr = tf.train.exponential_decay(
                self.config.lr_start, self.global_step, self.config.lr_decay_step,
                self.config.lr_decay_rate, staircase=True, name='lr')

        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
        self.summary_writer = None
        self.local_steps = 0

        self.build_summary()

    def start(self, sess, summary_writer):
        self.env.runner.start_runner(sess, self.local_network, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        rollout = self.env.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
          try:
            rollout.extend(self.env.runner.queue.get_nowait())
          except queue.Empty:
            break
        return rollout

    def process(self, sess):

        # 위에서 만든 operation self.sync를 run 시킨다.
        # worker network의 parameter를 global network의 parameter 로 만든다.

        sess.run(self.sync)
        rollout = self.pull_batch_from_queue()
        batch = self.process_rollout(rollout, gamma=0.99, lambda_= 1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
          fetches = [self.summary_op, self.train_op, self.global_step]
        else:
          fetches = [self.train_op, self.global_step]

        feed_dict = {
          self.local_network.x: batch.si,
          self.actions: batch.a,
          self.advantage: batch.adv,
          self.rewards: batch.r,
          self.local_network.state_in[0]: batch.features[0],
          self.local_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
          self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
          self.summary_writer.flush()
        self.local_steps += 1

    def build_summary(self):
        bs = tf.to_float(tf.shape(self.local_network.x)[0])

        tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
        tf.summary.scalar("model/value_loss", self.vf_loss / bs)
        tf.summary.scalar("model/entropy", self.entropy / bs)
        tf.summary.image("model/state", self.local_network.x)
        tf.summary.scalar("model/grad_global_norm", tf.global_norm(self.grads))
        tf.summary.scalar("model/var_global_norm", tf.global_norm(self.local_network.var_list))
        tf.summary.scalar("model/lr", self.lr)

        self.summary_op = tf.summary.merge_all()

    def process_rollout(self, rollout, gamma = 0.99, lambda_ = 1.0):
        raise Exception("Not implemented yet")