from ..rkbe import RKBEModel
import numpy as np
import tensorflow as tf

class RHHolE(RKBEModel):
	def __init__(self, n_entity=None,
		n_relation=None,
		entity_dim=50, 
		lambda_=None,
		lrate=.001,
		model_dir=None,
		dataName='DataUnknown',
		name=None,
		epoch_num=None,
		task=None,
		nozero=False ):
		if not name:
			name = 'RKBE%iD%sL.%s' % (entity_dim, str(lambda_) \
						if lambda_ else 'inf', dataName)
		relation_dim = h_dim = entity_dim; self.nozero = nozero
		super(RHHolE, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					task=task,
					lambda_=lambda_,
					lrate=lrate,
					name=name,
					model_dir=model_dir,
					epoch_num=epoch_num,
					dataName=dataName)
		
	def build_x(self):
		self.e1_fft = tf.fft(tf.complex(self.e1,0.0))
		self.e2_fft = tf.fft(tf.complex(self.e2,0.0))
		x = tf.multiply(self.r, tf.cast(tf.real(tf.ifft(tf.multiply(tf.conj(self.e1_fft),self.e2_fft))),dtype=tf.float32))
		return x


