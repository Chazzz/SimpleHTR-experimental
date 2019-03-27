from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import editdistance

import os
from os import path


class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model: 
	"minimalistic TF model for HTR"

	# model constants
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, model_name = 'default'):
		"init model: add CNN, RNN and CTC and initialize TF"
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.model_name = model_name
		self.snapID = 0

		self.is_train = tf.placeholder(tf.bool, name="is_train");

		# CNN
		self.inputImgs = tf.placeholder(tf.float32, shape=(Model.batchSize, Model.imgSize[0], Model.imgSize[1]))
		cnnOut4d = self.setupCNN(self.inputImgs)

		# RNN
		rnnOut3d = self.setupRNN(cnnOut4d)
		self.variable_summaries(rnnOut3d)

		# CTC
		(self.loss, self.decoder) = self.setupCTC(rnnOut3d)
		tf.summary.scalar('BIG loss', self.loss)

		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 

		# optimizer for NN parameters
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

		# initialize TF
		(self.sess, self.saver) = self.setupTF()

	def __del__(self):
		"""
		Since the buffers Model uses are (mostly) internal,
		it doesn't justify use of Python's with-as convention.
		Therefore del is used so Model cleans up after itself.
		"""
		self.train_writer.close()
		self.test_writer.close()

	def variable_summaries(self, var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

			
	def setupCNN(self, cnnIn3d):
		"create CNN layers and return output of these layers"
		cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

		# list of parameters for the layers
		kernelVals = [3, 3, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		# create layers
		pool = cnnIn4d # input to first CNN layer
		for i in range(numLayers):
			print(tf.shape(pool))
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			self.variable_summaries(conv)
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			kernel2 = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i+1], featureVals[i + 1]], stddev=0.1))
			conv2 = tf.nn.conv2d(relu, kernel2, padding='SAME',  strides=(1,1,1,1))
			self.variable_summaries(conv2)
			conv_norm2 = tf.layers.batch_normalization(conv2, training=self.is_train)
			relu2 = tf.nn.relu(conv_norm2)
			pool = tf.nn.max_pool(relu2, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

		return pool


	def setupRNN(self, rnnIn4d):
		"create RNN layers and return output of these layers"
		print(tf.shape(rnnIn4d))
		rnnIn3d = tf.squeeze(rnnIn4d, axis=[2])
		print(tf.shape(rnnIn3d))

		#50x32x236

		# basic cells which is used to build RNN
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# stack basic cells
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		# BxTxF -> BxTx2H
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
									
		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
									
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		return tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self, ctcIn3d):
		"create CTC loss and decoder and return them"
		# BxTxC -> TxBxC
		ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])
		# ground truth text as sparse tensor
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
		# calc loss for batch
		self.seqLen = tf.placeholder(tf.int32, [None])
		loss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True)
		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			decoder = tf.nn.ctc_beam_search_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			# import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
			word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

			# prepare information about language (dictionary, characters in dataset, characters forming words) 
			chars = str().join(self.charList)
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()

			# decode using the "Words" mode of word beam search
			decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

		# return a CTC operation to compute the loss and a CTC operation to decode the RNN output
		return (tf.reduce_mean(loss), decoder)


	def setupTF(self):
		"initialize TF"
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)

		sess=tf.Session() # TF session

		self.modelDir = '../model/'+self.model_name+'/'
		self.trainLogDir = '../logs/train/'+self.model_name
		self.testLogDir = '../logs/test/'+self.model_name
		for d in [self.modelDir, self.trainLogDir, self.testLogDir]:
			if not os.path.exists(d):
				os.mkdir(d)

		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(self.trainLogDir, sess.graph)
		self.test_writer = tf.summary.FileWriter(self.testLogDir) #TODO: Bring validation into tf
		# Use command 'tensorboard --logdir=../logs' to view saved logs

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		latestSnapshot = tf.train.latest_checkpoint(self.modelDir) # is there a saved model?
		# latestSnapshot = None #Never load snapshot

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput):
		"extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(Model.batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(Model.batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor 
			decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
			idxDict = { b : [] for b in range(Model.batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch, logging = True):
		"feed a batch into the NN to train it"
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
		if logging:
			(_, lossVal, summary, decoded) = self.sess.run([self.optimizer, self.loss, self.merged, self.decoder], { self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * Model.batchSize, self.learningRate : rate, self.is_train: True} )
			self.train_writer.add_summary(summary, self.batchesTrained)
			recognized = self.decoderOutputToText(decoded)
			numCharErr, numCharTotal, numWordOK, numWordTotal = self.calc_batch_stats(batch, recognized)
			charErrorRate = numCharErr / numCharTotal
			wordAccuracy = numWordOK / numWordTotal
			self.write_accuracy_summaries(self.train_writer, charErrorRate, wordAccuracy)
		else:
			(_, lossVal) = self.sess.run([self.optimizer, self.loss], { self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * Model.batchSize, self.learningRate : rate, self.is_train: True} )
		self.batchesTrained += 1
		return lossVal

	def calc_batch_stats(self, batch, recognized, verbose = False):
		numCharErr = 0
		numCharTotal = 0
		numWordOK = 0
		numWordTotal = 0
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			if verbose:
				print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
		return numCharErr, numCharTotal, numWordOK, numWordTotal

	def write_accuracy_summaries(self, writer, charErrorRate, wordAccuracy):
		"shared by test and validation to send comparable values to TensorBoard"
		summary = tf.Summary()
		summary.value.add(tag="wordAccuracy", simple_value=wordAccuracy)
		summary.value.add(tag="charErrorRate", simple_value=charErrorRate)
		writer.add_summary(summary, self.batchesTrained)

	def inferBatchStats(self, batch):
		"feed a batch into the NN to recognize the texts for validation"
		decoded = self.sess.run(self.decoder, { self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * Model.batchSize, self.is_train: False} )
		recognized = self.decoderOutputToText(decoded)
		return self.calc_batch_stats(batch, recognized)
	
	def inferBatch(self, batch):
		"feed a batch into the NN to recognize the texts"
		decoded = self.sess.run(self.decoder, { self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * Model.batchSize, self.is_train: False} )
		return self.decoderOutputToText(decoded)
	

	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
