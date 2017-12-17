import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface


class Adam(tf.train.AdamOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, loss, max_step=1000, verbose=0):
        """
        run optimizer to minimize loss

        :param loss: minimizing the loss for max_step steps

        :type loss: tf.Tensor

        :param max_step: maximum number of updates, default 1000

        :type max_step: int

        :param verbose: if 1, print more details, defaul  0

        :type verbose: int

        """
        if verbose: print('Minimizing Loss using Adam')
        strain_step = self.minimize(loss)
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        for step in range(max_step):
            sess.run(strain_step)
            if verbose and (step + 1) % 50 == 0:
                print("step {}\{}, loss {}".format(step + 1,
                                                   max_step,
                                                   loss.eval()))


class LBFGS():
    def __init__(self):
        self.__OptimizerWrap = \
            lambda loss, max_steps: ScipyOptimizerInterface(loss,
                                                            options={'maxiter': max_steps,
                                                                     'disp': 50})

    def train(self, loss, max_step=1000, verbose=0):
        """
                run optimizer to minimize loss

                :param loss: minimizing the loss for max_step steps

                :type loss: tf.Tensor

                :param max_step: maximum number of updates, default 1000

                :type max_step: int

                :param verbose: if 1, print more details, defaul  0

                :type verbose: int

        """
        if verbose: print('Minimizing Loss using L-BFGS')
        optimizer = self.__OptimizerWrap(loss, max_step)
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        optimizer.minimize(sess)
