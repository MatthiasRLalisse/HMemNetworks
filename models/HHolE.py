import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from ..base import KBEModel
from .WhitenEmbed import WhitenEmbed
from .WhitenEmbedSep import WhitenEmbedSep
no_zeros = 10e-10

def cconv(x, y):
	x_fft_ = tf.fft(tf.complex(x,0.0))
	#e2_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e2, axis=2),0.0))
	y_fft_ = tf.fft(tf.complex(y,0.0))
	x_fft = x_fft_ #+ tf.complex(tf.to_float(tf.equal(x_fft_, 0.)),0.)*no_zeros
	y_fft = y_fft_ #+ tf.complex(tf.to_float(tf.equal(y_fft_, 0.)),0.)*no_zeros
	return tf.cast(tf.real(tf.ifft(tf.multiply(tf.conj(x_fft),\
                                             y_fft))),dtype=tf.float32)

def ccorr(x, y):
	x_fft_ = tf.fft(tf.complex(x,0.0))
	#e2_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e2, axis=2),0.0))
	y_fft_ = tf.fft(tf.complex(y,0.0))
	x_fft = x_fft_ #+ tf.complex(tf.to_float(tf.equal(x_fft_, 0.)),0.)*no_zeros
	y_fft = y_fft_ #+ tf.complex(tf.to_float(tf.equal(y_fft_, 0.)),0.)*no_zeros
	return tf.cast(tf.real(tf.ifft(tf.multiply(x_fft,\
                                             y_fft))),dtype=tf.float32)



class HHolE(KBEModel):
	"""Harmonic Holographic Embedding class. Passes all arguments to the superclass KBEModel. 
	Compositional layer embeddings are given by:
		r * ifft( conj(fft(e1)) * fft(e2))) 	(* is elementwise multiplication)
	which is a computationally efficient formula for the circular convolution of vectors e1, e2.
	
	kwargs: n_entity=None,
		n_relation=None,
		entity_dim=50,
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
		Graph=None,
		entity_dim=50,
		task=None, 
		lambda_=None,
		max_neighbors=100,
		lrate=.001,
		gamma=0.,
		train_dropout=0.,
		model_dir=None,
		dataName='DataUnknown',
		name=None,
		epoch_num=None,
		trip=False, 
		whiten=False, 
		sep=True ):
		print('should whiten?', whiten)
		if not name: 
			name = 'HHolE%iD%sL%.2fdrop%s' % (entity_dim, str(lambda_) \
						if lambda_ else 'inf', train_dropout, \
						('andWH%s' % ('sep' if sep else '')) if whiten else '')	
							#sets naming convention for this model
			name += ('M'+ str(max_neighbors) if max_neighbors is not None 
						else (str(task.max_neighbors) if task is not None 
						else 'None') )
			if gamma: name += 'G%.4f' % (gamma)
			if trip: name += 'Trip'
			name += '-'+ (dataName if dataName != 'DataUnknown' else \
					(task.dataName if task is not None else 'DataUnKnown'))
		#print(name); print(task.dataName); print(dataName)
		relation_dim = h_dim = entity_dim
		self.whiten = whiten
		self.sep = sep
		print(name)
		print('@@@@@@@@@@@whitening is set to', self.whiten, whiten)
		if self.whiten:
			if self.sep:
				self.whitener = WhitenEmbedSep(h_dim)
			else:
				self.whitener = WhitenEmbed(h_dim)
			#self.get_embeddings = self.whitener.get_embeddings	
				#patch get_embeddings with whitening as post-process
		super(HHolE, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					model_class='HHolE', 
					Graph=Graph,
					task=task,
					lambda_=lambda_,
					gamma=gamma,
					train_dropout=train_dropout,
					max_neighbors=max_neighbors,
					normalize=False,
					lrate=lrate,
					name=name,
					model_dir=model_dir,
					epoch_num=epoch_num,
					dataName=dataName,
					trip=trip)
	def bind_op(self, r, e):
		return cconv(r, e)
	
	def unbind_op(self, Mem_h, r, e):	#0 is left direction, 1 is right, 
						#meaning we extract left (right) entities 
		#do any necessary memory tensor-reshaping here
		#right_probe = lambda: cconv(r, ccorr(e, Mem_h))
		#left_probe = lambda: cconv(r, cconv(e, Mem_h))
		probe = ccorr(r, Mem_h)
		return probe #tf.cond(self.probe_left, true_fn=left_probe, false_fn=right_probe)
	
	def build_probes(self):
		if self.whiten:
			#whiten_entities = [self.e1_embed, self.e2_embed, self.e_mem_embed ]
			#whiten_relations = [ self.r, self.r_mem_embed ]
			#Embeds = tf.concat([self.e_embeddings_, self.r_embeddings_], axis=0)
			e1_embed, e2_embed, e_mem_embed, r, r_mem_embed = \
				self.whitener.build_probes(self)
			print('@@@@@@@@@@@@@@build whitened embeddings')
				#self.whiteningTransform(Embeds, whiten_entities+whiten_relations)
			
		else:
			print('@@@@@@@@@@@@@@standard embeddings')
			e1_embed, e2_embed, e_mem_embed, r, r_mem_embed = \
					self.e1_embed, self.e2_embed, \
					self.e_mem_embed, self.r, self.r_mem_embed
		
		self.e_target = tf.cond(self.probe_left, true_fn=lambda:e1_embed, \
							false_fn=lambda:e2_embed)
		self.e_probe = tf.squeeze(tf.cond(self.probe_left, true_fn=lambda:e2_embed, \
							false_fn=lambda: e1_embed),1)
		self.e_mem_indices = tf.squeeze(tf.cond(self.probe_left, true_fn=lambda:self.e2_choice, \
							false_fn=lambda:self.e1_choice), axis=1)
	
	def mu_entities(self):
		raise NotImplementedError()



