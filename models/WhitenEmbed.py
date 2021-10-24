import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10

class WhitenEmbed(KBEModel):
	"""Class to whiten embeddings (ZCA transform)"""
	def __init__(self, h_dim):
		self.h_dim = h_dim
		#pass
	
	def whiteningTransform(self, Embeddings, embeddings_, alpha=.2, normalize=True):
		#E = tf.nn.l2_normalize(Embeddings)
		link = (lambda E: tf.nn.l2_normalize(E)) if normalize else (lambda E: E)
			#if normalize, then renormalize embeddings after whitening
		#E = tf.stop_gradient(Embeddings) #propagating through Cov leads to issues
		E = Embeddings #propagating through Cov leads to issues
		mean_e = tf.expand_dims(tf.reduce_mean(Embeddings, axis=0),0)
		E_centered = E - mean_e; I = tf.eye(self.h_dim)
		E_Cov_empirical = tf.matmul(E_centered, E_centered, \
					transpose_a=True)/tf.cast(tf.shape(E)[0], tf.float32)
		E_Cov = (1-alpha)*E_Cov_empirical + alpha*I
		S, U, _ = tf.linalg.svd(E_Cov)
		E_Prec_sqrt = tf.matmul(U, tf.matmul(tf.diag(1./tf.sqrt(S)), \
								U, transpose_b=True)) #multvar whitening
		E_stdev = tf.expand_dims(tf.sqrt(tf.reduce_mean(tf.square(E_centered), axis=0)),0) #univar whitening
		dim = tf.cast(tf.shape(E_centered)[-1], tf.float32)
		if type(embeddings_) is list:
			#out = [ link(e/(tf.sqrt(dim)*E_stdev)) for e in embeddings_ ]
			out = [ link(tf.tensordot(e, E_Prec_sqrt, axes=[-1,0])) for e in embeddings_ ]
		else: 
			#out = link(tf.tensordot(embeddings_, E_stdev, axes=[-1,0])/tf.sqrt(dim))
			out = link(tf.tensordot(embeddings_, E_Prec_sqrt, axes=[-1,0]))
		return out
	
	def build_probes(self, model):
		whiten_entities = [model.e1_embed, model.e2_embed, model.e_mem_embed ]
		whiten_relations = [ model.r, model.r_mem_embed ]
		Embeds = tf.concat([model.e_embeddings_, model.r_embeddings_], axis=0)
		e1_embed, e2_embed, e_mem_embed, r, r_mem_embed = \
			self.whiteningTransform(Embeds, whiten_entities+whiten_relations)
		return e1_embed, e2_embed, e_mem_embed, r, r_mem_embed


