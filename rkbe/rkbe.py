from ..base import KBETaskSetting, KBEModel
import numpy as np
from ..utils import permuteList
import tensorflow as tf
import sys

class RKBEModel(KBEModel):
	def __init__(self, 
		entity_dim=50, 
		relation_dim=None,
		h_dim=None,
		n_entity=None,
		n_relation=None,
		task=None,
		lambda_=None,
		lrate=.001,
		name='foo',
		model_dir='model', 
		epoch_num=None,
		dataName='DataUnKnown' ):	#model name allows restoring previous models
		if not name:
			name = 'RKBE%iD%sL.%s' % (entity_dim, str(lambda_) \
						if lambda_ else 'inf', dataName)
		super(RKBEModel, self).__init__(entity_dim=entity_dim,
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
	
	def build_embeddings(self):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.e_embeddings = tf.get_variable("entityEmbeddings", shape=[self.n_entity, self.entity_dim],
        			initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.r_embeddings = tf.get_variable("relationEmbeddings", shape=[self.n_relation, self.relation_dim],
        			initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.e1_choice = tf.placeholder(tf.int32, shape=[None, None])
		self.r_choice = tf.placeholder(tf.int32, shape=[None, None])
		self.e2_choice = tf.placeholder(tf.int32, shape=[None, None])

		e1_embed = tf.nn.embedding_lookup(self.e_embeddings, self.e1_choice)
		r_embed = tf.nn.embedding_lookup(self.r_embeddings, self.r_choice)
		e2_embed = tf.nn.embedding_lookup(self.e_embeddings, self.e2_choice)

		self.e1 = tf.nn.l2_normalize(e1_embed, axis=2)
		self.r = tf.nn.l2_normalize(r_embed, axis=2)
		self.e2 = tf.nn.l2_normalize(e2_embed, axis=2)


		
class RKBETaskSetting(KBETaskSetting):
	def __init__(self, *args, **kwargs):	#if True, candidate entities must satisfy relational type constraints
		super(RKBETaskSetting, self).__init__(*args, **kwargs)
		
	def trainLoop(self, model, e1choice, rchoice, e2choice, e1choice_neg, \
				rchoice_neg, e2choice_neg, sess=None):
		if not sess: sess = model.sess
		batch_size = len(rchoice)
		e1_choice_ = [ [e1choice[j]]*(self.negsamples+1) for j in range(batch_size) ]
		e1_choice_neg = [ [ e1choice[j] ] + e1choice_neg[j] for j in range(batch_size) ]
		r_choice_ = [ [rchoice[j]]*(self.negsamples+1) for j in range(batch_size) ] 
		r_choice_neg = [ [rchoice[j] ] + rchoice_neg[j] for j in range(batch_size) ]
		e2_choice_ = [ [e2choice[j]]*(self.negsamples+1)  for j in range(batch_size) ]
		e2_choice_neg = [ [ e2choice[j] ] + e2choice_neg[j] for j in range(batch_size) ]
		#train left entity
		batch_loss_left, null = sess.run([model.loss, model.train], {
					model.e1_choice: e1_choice_neg, 
					model.r_choice: r_choice_,
					model.e2_choice: e2_choice_})
		batch_loss_left = np.sum(batch_loss_left)
		#train right entity
		batch_loss_right, null = sess.run([model.loss, model.train], {
				model.e1_choice: e1_choice_, 
				model.r_choice: r_choice_, 
				model.e2_choice: e2_choice_neg })
		#train relation
		batch_loss_rel, null = sess.run([model.loss, model.train], {
				model.e1_choice: e1_choice_, 
				model.r_choice: r_choice_neg, 
				model.e2_choice: e2_choice_ })
		batch_loss_right = np.sum(batch_loss_right)
		return batch_loss_left + batch_loss_right + batch_loss_rel

	def trainEpoch(self, model, sess=None, interactive=False):
		if not sess: sess = model.sess
		epoch = model.epoch + 1
		e1s_train, rs_train, e2s_train = self.data['train'][:3]
		batches_ = int(len(e1s_train)/self.batch_size)
		perm_ = np.random.permutation(len(e2s_train))
		e1s_train_p, rs_train_p, e2s_train_p = [ permuteList(l, perm_) for l in \
							[e1s_train, rs_train, e2s_train] ]
		print('epoch {0}'.format(epoch))
		epoch_error = 0
		for i in range(batches_):
			e1choice, rchoice, e2choice = [ l[i*self.batch_size:i*self.batch_size + \
				self.batch_size] for l in [e1s_train_p, rs_train_p, e2s_train_p ] ]
			e1choice_neg = [ [ np.random.randint(self.n_entity) for n in range(self.negsamples) ] \
									for m in range(len(e1choice)) ]
			rchoice_neg = [ [ np.random.randint(self.n_relation) for n in range(self.negsamples) ] \
									for m in range(len(e2choice)) ]
			e2choice_neg = [ [ np.random.randint(self.n_entity) for n in range(self.negsamples) ] \
									for m in range(len(e2choice)) ]
			batch_loss = self.trainLoop(model, e1choice, rchoice, e2choice, \
						e1choice_neg, rchoice_neg, e2choice_neg, sess=sess)
			epoch_error += batch_loss
			if interactive: 
				sys.stdout.flush(); 
				sys.stdout.write(('\rtraining epoch %i \tbatch %i of %i \tbatch loss = %f\t\t'\
								% (epoch, i+1, batches_, batch_loss))+'\r')
		model.epoch += 1
		return epoch_error	
	
	def rankEntities(self, model, entity_1s,relations_,entity_2s, sess=None, direction='r'):
		if not sess: sess = model.sess
		true_triplets = (entity_1s,relations_,entity_2s)
		candidates_ = []
		entities_ = []; relations__ = []
		for j in range(len(entity_1s)):
			entity_1, relation_, entity_2 = entity_1s[j], relations_[j], entity_2s[j]
			if direction == 'r':	
				candidates = [ [entity_2] + [ e_ for e_ in self.data['candidates_r'][relation_] \
					if e_ != entity_2 and not(self.data['filter'][(entity_1,relation_,e_)]) ] ]
				entities_ += [[entity_1]*len(candidates) ]
			else:
				candidates = [[entity_1] + [ e_ for e_ in self.data['candidates_l'][relation_] \
					if e_ != entity_1 and not(self.data['filter'][(e_,relation_,entity_2)]) ] ]
				entities_ += [[entity_2]*len(candidates) ]
			relations__.append([relations_[j]]*len(candidates))
			candidates_ += candidates
		if direction=='r':
			scores = [sess.run( model.test_scores, {model.e1_choice: entities_,
								model.r_choice: relations__,
								model.e2_choice: candidates_ })]
		else:
			scores = [sess.run( model.test_scores, {model.e1_choice: candidates_,
								model.r_choice: relations__,
								model.e2_choice: entities_ })]
		candidates_perms = [ sorted( range(len(candidates)), key=lambda x:scores[j][x] )[::-1] \
							for j,candidates in enumerate(candidates_) ]
		ranked = [ [ candidates[i] for i in candidates_perms[j] ] for j,candidates in enumerate(candidates_) ]
		return ranked
	
	def rankRelations(self, model, entity_1s,relations_,entity_2s, sess=None):
		if not sess: sess = model.sess
		true_triplets = (entity_1s,relations_,entity_2s)
		candidates_ = []
		entity1s_ = []; entity2s_ = []
		for j in range(len(entity_1s)):
			entity_1, relation_, entity_2 = entity_1s[j], relations_[j], entity_2s[j]
			candidates = [[relation_] + [ r_ for r_ in range(self.n_relation) if r_ != relation_ \
						and not(self.data['filter'][(entity_1,r_,entity_2)]) ]]
			entity1s_ += [[entity_1]*len(candidates) ]
			entity2s_ += [[entity_2]*len(candidates) ]
			candidates_ += candidates
		scores = [sess.run( model.test_scores, {model.e1_choice: entity1s_,
								model.r_choice: candidates_,
								model.e2_choice: entity2s_ })]
		candidates_perms = [ sorted( range(len(candidates)), key=lambda x:scores[j][x] )[::-1] \
							for j,candidates in enumerate(candidates_) ]
		ranked = [ [ candidates[i] for i in candidates_perms[j] ] for j,candidates in enumerate(candidates_) ]
		return ranked
	
	def rank(self, model, entity_1, relation_, entity_2, sess=None, arg=0):
		assert arg in range(3)
		if not sess: sess = model.sess
		direction = 'l' if arg == 0 else ('r' if arg == 2 else None)
		ranked = self.rankEntities(model, entity_1, relation_, entity_2, \
						sess=sess, direction=direction) \
			if arg != 1 else \
				self.rankRelations(model, entity_1, relation_, entity_2, sess=sess)
		if arg==0:
			rank = [ ranks_.index(entity_1[j])+1 for j, ranks_ in enumerate(ranked) ]
		elif arg==1:
			rank = [ ranks_.index(relation_[j])+1 for j, ranks_ in enumerate(ranked) ]
		else:
			rank = [ ranks_.index(entity_2[j])+1 for j,ranks_ in enumerate(ranked) ]
		return rank

	def eval(self, model, sess=None, test_set=False, interactive=False):
		if not sess: sess = model.sess
		datatype = 'test' if test_set else 'valid'
		print('testing...\n') 
		eval_data = self.data[datatype]
		e1s_test, rs_test, e2s_test = eval_data[:3]
		test_batch_size = 1
		perm_ = np.random.permutation(len(e2s_test))
		e1s_test_p, rs_test_p, e2s_test_p = [ permuteList(l, perm_) for l in [e1s_test,rs_test,e2s_test] ]
		test_batches_ = int(np.ceil(len(e1s_test_p)/float(test_batch_size)))
		n_examples = 0; ranks_left = []; ranks_right = []; ranks_rel = []
		hits_1l = 0.; hits_3l = 0.; hits_10l = 0.
		hits_1r = 0.; hits_3r = 0.; hits_10r = 0.
		hits_1rel = 0.; hits_3rel = 0.; hits_10rel=0.
		for k in range(test_batches_):
			c = k-1
			e1_, r_, e2_ = [ l[k*test_batch_size:k*test_batch_size + test_batch_size] \
							for l in [e1s_test_p, rs_test_p, e2s_test_p ] ]
			n_examples += len(e1_)
			right_rank = self.rank(model, e1_,r_,e2_, sess=sess, arg=2)
			right_rank_arr = np.array(right_rank,dtype=np.int32)
			hits_1r += np.sum(right_rank_arr == 1)
			hits_3r += np.sum(right_rank_arr <= 3)
			hits_10r += np.sum(right_rank_arr <= 10)
			left_rank = self.rank(model, e1_,r_,e2_, sess=sess, arg=0)
			left_rank_arr = np.array(left_rank)
			hits_1l += np.sum(left_rank_arr == 1)
			hits_3l += np.sum(left_rank_arr <= 3)
			hits_10l += np.sum(left_rank_arr <= 10)
			rel_rank = self.rank(model, e1_,r_,e2_, sess=sess, arg=1)
			rel_rank_arr = np.array(rel_rank,dtype=np.int32)
			hits_1rel += np.sum(rel_rank_arr == 1)
			hits_3rel += np.sum(rel_rank_arr <= 3)
			hits_10rel += np.sum(rel_rank_arr <= 10)
			ranks_right += right_rank
			ranks_left += left_rank
			ranks_rel += rel_rank
			mean_rank_e1 = np.sum(left_rank_arr)/float(len(left_rank))
			mean_rank_e2 = np.sum(right_rank_arr)/float(len(right_rank))
			mean_rank_rel = np.sum(rel_rank_arr)/float(len(rel_rank))
			MRR_left = np.sum([ 1./rank_ for rank_ in ranks_left ])/len(ranks_left)
			MRR_right = np.sum([ 1./rank_ for rank_ in ranks_right ])/len(ranks_right)
			MRR_rel = np.sum([1./rank_ for rank_ in ranks_rel ])/len(ranks_rel)
			if interactive: 
				sys.stdout.flush()
				sys.stdout.write('\r\tbatch %i of %i: rank(e1) = %i \trank(r) = %i \trank(e2) = %i '\
					'MRR = %.5f, Hits@1 = %.5f, Hits@3 = %.5f, '\
					'Hits@10 = %.5f\t\t\r' % \
					(k+1, test_batches_, mean_rank_e1, mean_rank_rel, mean_rank_e2, \
					(MRR_left + MRR_right + MRR_rel)/3., \
					(hits_1l + hits_1r + hits_1rel)/(n_examples*3), \
					(hits_3l + hits_3r + hits_3rel)/(n_examples*3), \
					(hits_10l + hits_10r + hits_10rel)/(n_examples*3)))
		if interactive: print('\n')
		mean_rank_left = np.sum(ranks_left)/float(len(ranks_left))
		mean_rank_right = np.sum(ranks_right)/float(len(ranks_right))
		mean_rank_rel = np.sum(ranks_rel)/float(len(ranks_rel))
		results = {	'MR_left':mean_rank_left,
				'MR_right':mean_rank_right,
				'MR_rel':mean_rank_rel,
				'MRR_left':MRR_left,
				'MRR_right':MRR_right,
				'MRR_rel':MRR_rel,
				'Hits1_left':hits_1l/n_examples,
				'Hits3_left':hits_3l/n_examples,
				'Hits10_left':hits_10l/n_examples,
				'Hits1_right':hits_1r/n_examples,
				'Hits3_right':hits_3r/n_examples,
				'Hits10_right':hits_10r/n_examples,
				'Hits1_rel':hits_1rel/n_examples,
				'Hits3_rel':hits_3rel/n_examples,
				'Hits10_rel':hits_10rel/n_examples,
				'Hits10': (hits_10l + hits_10r + hits_10rel)/(n_examples*3),
				'Hits3': (hits_3l + hits_3r + hits_3rel)/(n_examples*3),
				'Hits1': (hits_1l + hits_1r + hits_10rel)/(n_examples*3),
				'MRR': (MRR_left + MRR_right + MRR_rel)/3.,
				'MR': (mean_rank_left + mean_rank_right + mean_rank_rel)/3. }
		model.results[(model.epoch, datatype)] = results
		return results

