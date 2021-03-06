import hmem as hm
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from optparse import OptionParser

config = tf.ConfigProto() 
config.gpu_options.allow_growth = True

parser = OptionParser()
parser.add_option("-d", "--data", action="store", type="string", default='freebase', dest="data")
parser.add_option("-N", "--entity_dim", action="store", type="int", default=50, dest="entity_dim")
parser.add_option("-R", "--relation_dim", action="store", type="int", default=20, dest="relation_dim")
parser.add_option("-b", "--batch_size", action="store", type="int", default=512, dest="batch_size")
parser.add_option("-M", "--negsamples", action="store", type="int", default=500, dest="negsamples")
parser.add_option("-i", "--lambda", action="store", type="float", default=0.0, dest="lambda_")
parser.add_option("-T", "--test_every", action="store", type="int", default=1, dest="test_every")
parser.add_option("-I", "--interactive", action="store_true", default=False, dest="interactive")
parser.add_option("-e", "--epochs", action="store", type="int", default=100, dest="epochs")
parser.add_option("-r", "--dropout", action="store", type="float", default=0., dest="train_dropout")
parser.add_option("-n", "--num_to_test", action="store", type="int", default=0, dest="num_to_test")
parser.add_option("-m", "--max_neighbors", action="store", type="int", default=100, dest="max_neighbors")
parser.add_option("-t", "--task_max_neighbors", action="store", type="int", default=800, dest="task_max_neighbors")
parser.add_option("-C", "--model_class", action="store", type="str", default="HHolE", dest="model_class")
parser.add_option("-E", "--min_epochs", action="store", type="int", default=15, dest="min_epochs")
parser.add_option("-S", "--test", action="store_true", default=False, dest="test")
parser.add_option("-F", "--full_graph", action="store_true", default=False, dest="full_graph")
parser.add_option("-W", "--whiten", action="store_true", default=False, dest="whiten")

(options, args) = parser.parse_args()
MAXEPOCH = 500; MINEPOCH = options.min_epochs
TEST_EVERY = options.test_every; INTERACTIVE = options.interactive
DIM = options.entity_dim; RELATION_DIM=options.relation_dim
BATCHSIZE = options.batch_size; NEGSAMPLES = options.negsamples; 
DATA = options.data
LAMBDA = options.lambda_
TRAIN_DROPOUT = options.train_dropout
NUM_TO_TEST = options.num_to_test
MAX_NEIGHBORS = options.max_neighbors
TASK_MAX_NEIGHBORS = options.task_max_neighbors
MODELCLASS = options.model_class
TEST = options.test
FULL_GRAPH = options.full_graph
WHITEN = options.whiten

print('options say to whiten', WHITEN)

#np.random.seed(575389)

task = hm.KBETaskSetting(dataName=DATA, negsamples=NEGSAMPLES, batch_size=BATCHSIZE, \
			max_neighbors=TASK_MAX_NEIGHBORS, type_constrain=True, filtered=True )
if MODELCLASS == 'HHolE':
	model = hm.models.HHolE(task=task, entity_dim=DIM, lambda_=LAMBDA, whiten=WHITEN, 
			train_dropout=TRAIN_DROPOUT, max_neighbors=MAX_NEIGHBORS)
elif MODELCLASS == 'HTPR': 
	model = hm.models.HTPR(task=task, entity_dim=DIM, relation_dim=RELATION_DIM, 
			lambda_=LAMBDA, train_dropout=TRAIN_DROPOUT, max_neighbors=MAX_NEIGHBORS)
elif MODELCLASS == 'BilinBinding': 
	model = hm.models.BilinBinding(task=task, entity_dim=DIM, relation_dim=RELATION_DIM, 
			lambda_=LAMBDA, train_dropout=TRAIN_DROPOUT, max_neighbors=MAX_NEIGHBORS)
elif MODELCLASS == 'BilinBindingWithCov': 
	model = hm.models.BilinBinding(task=task, entity_dim=DIM, relation_dim=RELATION_DIM, 
			lambda_=LAMBDA, train_dropout=TRAIN_DROPOUT, max_neighbors=MAX_NEIGHBORS)


results = hm.utils.read_results('results/'+model.name+'-results.txt')
epochwise_accuracy = results['MRR']

for e in range(len(epochwise_accuracy)-1, MAXEPOCH+1):	
	if TEST:
	  break
	task.trainEpoch(model, interactive=INTERACTIVE)	#train the model for one epoch
	print('\n\n')
	if e % TEST_EVERY == 0: 
		print('epoch %i\n'%e)
		results_valid = task.eval(model, interactive=INTERACTIVE, num_to_test=NUM_TO_TEST) 	
									#evaluate on validation set
		newline = 'epoch ' + str(model.epoch) + '\t' + \
					'\t'.join( [ key + ' ' + str(results_valid[key]) \
					for key in [ 'MR', 'MRR', 'Hits1', 'Hits3', 'Hits10' ] ]) + '\n'
		acc = results_valid['MRR']; epochwise_accuracy.append(acc)
		keep_training = ( e <= MINEPOCH or acc > np.min(epochwise_accuracy[-10:]) )	
						#train at least 5 epochs, and until performance
						#is lower than the moving average
		with open('results/' + model.name + '-results.txt', 'a') as f:
			f.write(newline)
		if acc >= np.max(epochwise_accuracy):
			model.save()
		if not keep_training: break

best_epoch = epochwise_accuracy.index(max(epochwise_accuracy)) + 1
tf.reset_default_graph()

if MODELCLASS == 'HHolE':
	test_model = hm.models.HHolE(task=task, entity_dim=DIM, lambda_=LAMBDA, whiten=WHITEN,  
			train_dropout=TRAIN_DROPOUT, max_neighbors=MAX_NEIGHBORS)
elif MODELCLASS == 'HTPR': 
	test_model = hm.models.HTPR(task=task, entity_dim=DIM, relation_dim=RELATION_DIM, 
			lambda_=LAMBDA, train_dropout=TRAIN_DROPOUT, max_neighbors=MAX_NEIGHBORS)
elif MODELCLASS == 'BilinBinding': 
	test_model = hm.models.BilinBinding(task=task, entity_dim=DIM, relation_dim=RELATION_DIM, 
			lambda_=LAMBDA, train_dropout=TRAIN_DROPOUT, max_neighbors=MAX_NEIGHBORS)


results_test = task.eval(test_model, test_set=True, full_graph=FULL_GRAPH, interactive=INTERACTIVE) 	#evaluate on test set

print('\n')
print(results_test['Hits1'], results_test['Hits3'], results_test['Hits10'])

newline = '\nTEST\tepoch ' + str(test_model.epoch) + '\t' + '\t'.join( [ key + ' ' + str(results_test[key]) \
						for key in [ 'MR', 'MRR', 'Hits1', 'Hits3', 'Hits10' ] ])
with open('results/' + test_model.name + '-results.txt', 'a') as f:
	f.write(newline)






