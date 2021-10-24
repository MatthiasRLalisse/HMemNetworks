import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10

class WhitenEmbed(KBEModel):
	"""Class to whiten embeddings (ZCA transform)"""
	def __init__(self):
		pass
	def get_embeddings(self, Embeddings, embeddings_, alpha=.2):
		#E = tf.nn.l2_normalize(Embeddings)
		E = tf.stop_gradient(Embeddings) #propagating through Cov leads to issues
		mean_e = tf.expand_dims(tf.reduce_mean(Embeddings, axis=0),0)
		E_centered = E - mean_e; I = tf.eye(self.h_dim)
		E_Cov_empirical = tf.matmul(E_centered, E_centered, \
					transpose_a=True)/tf.cast(tf.shape(E)[0], tf.float32)
		E_Cov = (1-alpha)*E_Cov_empirical + alpha*I
		S, U, _ = tf.linalg.svd(E_Cov)
		E_Prec_sqrt = tf.matmul(U, tf.matmul(tf.diag(1./tf.sqrt(S)), \
								U, transpose_b=True))
		E_stdev = tf.expand_dims(tf.sqrt(tf.reduce_mean(tf.square(E_centered), axis=0)),0)
		#self.E_Prec_sqrt = tf.diag(1./E_stdev)
		dim = tf.cast(tf.shape(E_centered)[-1], tf.float32)
		#print(embeddings_)
		if type(embeddings_) is list:
			out = [ e/(tf.sqrt(dim)*E_stdev) for e in embeddings_ ]
		else: 
			out = tf.tensordot(embeddings_, self.E_Prec_sqrt, axes=[-1,0])/dim
		return out



