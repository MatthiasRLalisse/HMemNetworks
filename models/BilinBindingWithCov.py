import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from ..base import KBEModel

eps = 1e-7

def pinv(a, rcond=1e-15):
    s, u, v = tf.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.greater(s, limit)
    
    reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(s.shape, dtype=a.dtype))
    lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
    return tf.matmul(lhs, u, transpose_b=True)

mm = tf.linalg.matmul
inv = tf.linalg.inv
trans = tf.transpose

class BilinBinding(KBEModel):
	"""Harmonic Tensor Product Representation class. Passes all arguments to the superclass KBEModel. 
	Compositional layer embeddings are given by the tensor product of e1, r, and e2, with dimensionality
	dim(e1)*dim(r)*dim(e2). 
	kwargs: n_entity=None,
		n_relation=None,
		entity_dim=5,
		task=None, 
		lambda_=None,
		lrate=.001,
		model_dir='trained_models',
		dataName='DataUnknown',
		name=None,
		epoch_num=None"""
	def __init__(self,
		n_entity=None,
		n_relation=None,
		entity_dim=50, 
		relation_dim=15,
		h_dim=None,
		task=None,
		lambda_=None,
		gamma=0.0,
		train_dropout=0.,
		max_neighbors=100,
		lrate=.001,
		model_dir=None,
		dataName='DataUnknown',
		name=None,
		epoch_num=None ):
		if not relation_dim: relation_dim = entity_dim
		if task is not None: dataName = task.dataName
		if not name:
			name = 'BilinBindingWithCov%ieD%irD%sL-%s' % (entity_dim, relation_dim, str(lambda_) \
							if lambda_ else 'inf', dataName)
			if gamma: name += 'G%.3f' % (gamma)
			name += ('M'+ str(max_neighbors) if max_neighbors is not None 
						else (str(task.max_neighbors) if task is not None 
						else 'None') )
			self.name = name
		h_dim = max(entity_dim, relation_dim) if h_dim is None else h_dim
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.WBind = tf.get_variable("Weights_Bind", 
					shape=[h_dim, relation_dim, entity_dim],
					dtype=tf.float32)
			self.XCov_ = tf.get_variable("Weights_TPRCov", 
					shape=[relation_dim*entity_dim, relation_dim*entity_dim], 
					dtype=tf.float32)
		self.XCov = mm(self.XCov_, trans(self.XCov_.T)) + eps*tf.eye(entity_dim*relation_dim)
		super(BilinBinding, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					model_class=BilinBinding,
					task=task,
					lambda_=lambda_,
					gamma=gamma,
					train_dropout=train_dropout,
					max_neighbors=max_neighbors,
					normalize=True,
					lrate=lrate,
					name=name, 
					dataName = dataName,
					model_dir=model_dir,
					epoch_num=epoch_num)
		#self.mu_h_1, self.mu_h_2 = self.mu_entities()
		print(self.WBind)
	def bind_op(self, r, e):
		#tpr = tf.einsum('bmi,bmj->bmij', r, e)
		h_vec = tf.einsum('hij,bmi,bmj->bmh', self.WBind, r, e)
		#batchdim = tf.cast(tf.shape(e)[0], tf.int32)
		#memdim = tf.cast(tf.shape(e)[1], tf.int32)
		#unravelled_tpr = tf.reshape(tpr, \
		#			[batchdim, memdim, self.h_dim])
		#print(unravelled_tpr.shape, r.shape, e.shape)
		return h_vec #unravelled_tpr
	
	def unbind_op(self, Mem_h, r, e):	#0 is left direction, 1 is right, 
						#meaning we extract left (right) entities 
		#do any necessary memory tensor-reshaping here
		batchdim = tf.cast(tf.shape(Mem_h)[0], tf.int32)
		WBind_ = tf.reshape(self.WBind, [self.h_dim, -1])
		X = self.XCov
		self.WUBind = mm(X, mm(trans(WBind_), inv(mm(WBind, mm(X, trans(WBind))))))
		tpr_ravel = tf.einsum('kh,bh->bk', self.WUBind, Mem_h)
		self.mu_M_tpr = tf.reshape(tpr_ravel, \
		                   [batchdim, self.relation_dim, self.entity_dim])
		probe = tf.einsum('bi,bij->bj', r, self.mu_M_tpr)
		return probe #tf.cond(self.probe_left, true_fn=left_probe, false_fn=right_probe)
	
	def score(self):
		scores = tf.reduce_sum(tf.multiply(tf.expand_dims(self.e_out,1),self.e_target), axis=-1)
		return scores	
	
	def mu_entities(self):
		raise NotImplementedError()



