import sys
import time
import numpy as np
import tensorflow as tf
from noise import update_noise_rates, get_noise_matrix
from .model_spec import AlexNet


class ConvNet(object):
    def __init__(self, model_name, train_batch_size, test_batch_size, robust=None, rho=None, classes=2,
                 eval_frequency=100, num_epochs=100, train_size=2048, test_size=2000, lambda_reg=5e-4,
                 base_lr=0.01, decay_rate=0.95, momentum=0.9):
        self.rho = rho
        self.model_name = model_name
        self.robust = robust
        self.classes = classes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.eval_frequency = eval_frequency
        self.num_epochs = num_epochs
        self.train_size = train_size
        self.test_size = test_size
        self.lambda_reg = lambda_reg
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.momentum = momentum

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.logit_model = AlexNet() if model_name == 'alexnet' else None
            self._create_model()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.graph_init)
        self.saver.save(self.sess, self.logit_model.checkpoint_file)

    def _create_model(self):
        self.noise_matrix = tf.placeholder(tf.float32, shape=(2, 2), name='noise_matrix')
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        input_shape = (self.logit_model.width, self.logit_model.height, self.logit_model.num_channels)
        self.train_data_node = tf.placeholder(tf.float32, shape=(None,) + input_shape)
        self.train_labels_node = tf.placeholder(tf.int64, shape=(None,))
        self.train_indicators_node = tf.placeholder(tf.int64, shape=(None,))
        self.eval_data = tf.placeholder(tf.float32, shape=(None,) + input_shape)

        # Training computation: logits + cross-entropy loss.
        self.logits = self.logit_model.logit(self.train_data_node)
        self.eval_prediction = tf.nn.softmax(self.logit_model.logit(self.eval_data))

        # Predictions for the current training minibatch.
        self.train_prediction = tf.nn.softmax(self.logits)

        if self.robust in ['robust_ml', 'robust_em', 'robust_map']:
            self.loss_unreg = self._robust_loss(self.logits, self.train_labels_node,
                                                self.noise_matrix, self.train_indicators_node)
        else:
            self.loss_unreg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                      labels=self.train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = tf.add_n([tf.nn.l2_loss(w) for w in self.logit_model.weights])
        # Add the regularization term to the loss.
        self.loss = self.loss_unreg + self.lambda_reg * regularizers
        self._setup_optimizer(train_size=self.train_size)
        self.graph_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)
        tf.get_default_graph().finalize()  # graph is constructed, make it readonly from here on

    def _setup_optimizer(self, train_size):
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0, dtype=tf.float32, trainable=False)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
            self.base_lr,  # Base learning rate.
            batch * self.train_batch_size,  # Current index into the dataset.
            train_size,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.gradient = [tf.reshape(g[0], [-1]) for g in optimizer.compute_gradients(self.loss) if g[0] is not None]
        self.minimizer = optimizer.minimize(self.loss, global_step=batch)

    def _robust_loss(self, logits, train_labels_node, K_tf, train_indicators_node):
        qb = tf.cast(train_indicators_node, tf.bool)
        qnb = tf.logical_not(qb)
        logits_clean = tf.boolean_mask(logits, qb)
        labels_clean = tf.boolean_mask(train_labels_node, qb)
        labels_noise = tf.boolean_mask(train_labels_node, qnb)
        logits_masked = tf.boolean_mask(logits, qnb)
        lse = tf.reduce_logsumexp(logits_masked, axis=1, keep_dims=True)

        Ks = tf.gather(tf.transpose(K_tf), labels_noise)
        logits_noise = tf.where(tf.equal(Ks, 0), -np.inf * tf.ones_like(Ks), logits_masked)
        lmax = tf.stop_gradient(tf.reduce_max(logits_noise, axis=1, keep_dims=True))
        wlse = lmax + tf.log(tf.reduce_sum(tf.multiply(Ks, tf.exp(logits_noise - lmax)), axis=1, keep_dims=True))
        self.loss_noise = tf.reduce_sum(lse - wlse)
        self.loss_clean = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_clean, labels=labels_clean))
        loss = (self.loss_noise + self.loss_clean) / tf.constant(self.train_batch_size, dtype=tf.float32)
        return loss

    def fit(self, X, y, q=None, beta=None, noise_batch_size=256):
        # label noise indicator is encoded as
        # q = 1: clean, q = 0: noisy

        train_data = X
        train_labels = y
        train_indicators = np.ones(len(y)) if q is None else q
        train_size = len(y)

        train_noisy_labels = train_labels[train_indicators == 0]
        train_noisy_data = train_data[train_indicators == 0, :]
        noisy_labels_size = len(train_noisy_labels)

        # Create a local session to run the training.
        start_time = time.time()
        self.saver.restore(self.sess, self.logit_model.checkpoint_file)

        # Run all the initializers to prepare the trainable parameters.
        print('Initialized!')
        # Loop through training steps.
        for step in range(int(self.num_epochs * train_size) // self.train_batch_size):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * self.train_batch_size) % train_size
            batch_data = train_data[offset:(offset + self.train_batch_size), ...]
            batch_labels = train_labels[offset:(offset + self.train_batch_size)]
            batch_indicators = train_indicators[offset:(offset + self.train_batch_size)]
            # This dictionary maps the batch data (as a np array) to the
            # node in the graph it should be fed to.
            feed_dict = {self.train_data_node: batch_data,
                         self.train_labels_node: batch_labels,
                         self.train_indicators_node: batch_indicators,
                         self.noise_matrix: get_noise_matrix(self.rho)}

            # Run the optimizer to update weights.
            self.sess.run(self.minimizer, feed_dict=feed_dict)

            if noisy_labels_size > 0 and self.robust in ['robust_em', 'robust_map']:
                noise_batch_size = min(noise_batch_size, noisy_labels_size)

                offset_noise = (step * noise_batch_size) % noisy_labels_size
                batch_noisy_data = train_noisy_data[offset_noise:offset_noise + noise_batch_size, :]
                batch_noisy_labels = train_noisy_labels[offset_noise:offset_noise + noise_batch_size]
                batch_noisy_indicators = np.zeros((len(batch_noisy_labels)))

                pred = self.predict_proba_train(batch_noisy_data)[:, 1]
                b = np.ones((2, 2)) if beta is None else beta
                self.rho = update_noise_rates(pred, batch_noisy_labels, self.rho, batch_noisy_indicators, b)

            # print some extra information once reach the evaluation frequency
            if step % self.eval_frequency == 0:
                # fetch some extra nodes' data
                l, lr, predictions = self.sess.run([self.loss, self.learning_rate, self.train_prediction],
                                              feed_dict=feed_dict)

                elapsed_time = time.time() - start_time
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * self.train_batch_size / train_size,
                       1000 * elapsed_time / self.eval_frequency))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % self.error_rate(predictions, batch_labels))
                sys.stdout.flush()
        return self

    def error_rate(self, predictions, labels):
        """Return the error rate based on dense predictions and sparse labels."""
        return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

    def score(self, X, y):
        test_accuracy = 1. - self.error_rate(self.predict_proba(X), y) * 0.01
        return test_accuracy

    def predict_proba(self, X):
        return self.eval_in_batches(X, self.test_batch_size, self.eval_prediction, self.eval_data)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)

    def predict_proba_train(self, X):
        return self.eval_in_batches(X, self.train_batch_size, self.train_prediction, self.train_data_node)

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(self, data, batch_size, op, node):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < batch_size:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, self.classes), dtype=np.float32)
        for begin in range(0, size, batch_size):
            end = begin + batch_size
            if end <= size:
                predictions[begin:end, :] = self.sess.run(op, feed_dict={node: data[begin:end, ...]})
            else:
                batch_predictions = self.sess.run(op, feed_dict={node: data[-batch_size:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        predictions = self.sess.run(self.eval_prediction, feed_dict={self.eval_data: data})
        return predictions

    def robust_gradient_diff_norm(self, Sx, Sy_noise, q):
        return self._diff_norm(Sx, Sy_noise, q=q)

    def gradient_diff_norm(self, Sx, Sy_noise):
        return self._diff_norm(Sx, Sy_noise, q=np.ones_like(Sy_noise))

    def _diff_norm(self, Sx, Sy_noise, q):
        train_data = Sx
        train_labels = Sy_noise
        train_indicators = q
        train_size = len(train_labels)
        df = np.zeros((train_size, self.classes))

        for step in range(train_size // self.train_batch_size):
            offset = (step * self.train_batch_size) % (train_size - self.train_batch_size)
            batch_data = train_data[offset:(offset + self.train_batch_size), ...]
            batch_labels = train_labels[offset:(offset + self.train_batch_size)]
            batch_indicators = train_indicators[offset:(offset + self.train_batch_size)]

            feed_dict = {self.train_data_node: batch_data,
                         self.train_labels_node: batch_labels,
                         self.train_indicators_node: batch_indicators,
                         self.noise_matrix: get_noise_matrix(self.rho)}

            gradval = np.concatenate(self.sess.run(self.gradient, feed_dict=feed_dict))

            for j in range(self.train_batch_size):
                for k in range(self.classes):
                    batch_labels_k = batch_labels.copy()
                    batch_labels_k[j] = k
                    q_k = batch_indicators.copy()
                    q_k[j] = 1
                    feed_dict[self.train_indicators_node] = q_k
                    feed_dict[self.train_labels_node] = batch_labels_k
                    gradval_jk = np.concatenate(self.sess.run(self.gradient, feed_dict=feed_dict))
                    df[offset + j, k] = np.linalg.norm(gradval - gradval_jk, 2)
            print('step = %d' % step)
        return df
