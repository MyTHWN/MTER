# Tensor Multi-task 9/11/2017 NanWang

import os
import math
import random
import pickle
import argparse

import numpy as np
from sklearn.cluster import KMeans

import multi_tasks_paraserver as tsmtr


def load_useritemfea(file, U_num, I_num):
	sps_tensor_useritemf = {}
	useritem_ls = []
	overall_rating = np.zeros((U_num,I_num))
	
	with open(file, 'r') as fin:
		lines = fin.readlines()
		
		for line in lines:
			eachline = line.strip().split(',')
			ft_sent_pair = eachline[3].strip()
			
			if ft_sent_pair != '':
				u_idx = int(eachline[0])
				i_idx = int(eachline[1])
				if (u_idx, i_idx) not in useritem_ls:
					useritem_ls.append((u_idx, i_idx))
				over_rating = int(eachline[2])
				if over_rating == 0:
					over_rating = 1
				f_s_pairs = ft_sent_pair.strip().split(' ')
				for f_s in f_s_pairs:
					fea = f_s.strip().split(':')
					f_idx = int(fea[0])
					senti = int(fea[1])
					if (u_idx,i_idx,f_idx) not in sps_tensor_useritemf:
						sps_tensor_useritemf[(u_idx,i_idx,f_idx)] = 0			
					sps_tensor_useritemf[(u_idx,i_idx,f_idx)] += senti 
					#if item_feature_mentioned[i_idx][f_idx] == 0:
						#item_feature_num_record[i_idx] += 1
					#item_feature_mentioned[i_idx][f_idx] += 1
				overall_rating[u_idx,i_idx] = over_rating
				sps_tensor_useritemf[(u_idx,i_idx,F_num)] = over_rating
	
	return overall_rating, sps_tensor_useritemf, useritem_ls


def load_useritemfeaword(file, F_num, W_num):
	sps_tensor_userwordf = {}
	sps_tensor_itemwordf = {}
	feature_word_used = np.zeros((F_num,W_num))
	
	with open(file, 'r') as fin:
		lines = fin.readlines()

	for line in lines:
		eachline = line.strip().split(',')
		u_idx = int(eachline[0])
		i_idx = int(eachline[1])
		f_idx = int(eachline[2])
		w_idx = int(eachline[3])
		feature_word_used[f_idx][w_idx] += 1

		if (u_idx,f_idx,w_idx) not in sps_tensor_userwordf:
			sps_tensor_userwordf[(u_idx,f_idx,w_idx)] = 0
		sps_tensor_userwordf[(u_idx,f_idx,w_idx)] += 1

		if (i_idx,f_idx,w_idx) not in sps_tensor_itemwordf:
			sps_tensor_itemwordf[(i_idx,f_idx,w_idx)] = 0
		sps_tensor_itemwordf[(i_idx,f_idx,w_idx)] += 1

	return sps_tensor_userwordf, sps_tensor_itemwordf, feature_word_used 


if __name__ == '__main__':    
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-u', '--useremb', default=15, type=int, help='user embedding dimension')
	parser.add_argument('-i', '--itememb', default=15, type=int, help='item embedding dimension')
	parser.add_argument('-f', '--featemb', default=12, type=int, help='feature embedding dimension')
	parser.add_argument('-w', '--wordemb', default=12, type=int, help='opinion embedding dimension')
	
	parser.add_argument('--bpr', default=10, type=float, help='weight of BPR loss')
	parser.add_argument('--iter', default=200000, type=int, help='number of training iterations')	
	parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
	
	parser.add_argument('--nprocess', default=4, type=int, help='number of processes for multiprocessing')

	args = parser.parse_args()
	
	outfile = 'mter_paraserver'
	U_num = 10719
	I_num = 10410
	F_num = 104  #+1 = 105 including overall rating
	W_num = 1019

	print('Preparing Data...')
	#load the interactions between user, item and feature
	overall_rating_trn, sps_tensor_useritemf_trn, useritem_ls_trn = \
			load_useritemfea('yelp_recursive_entry/yelp_recursive_train.entry', U_num, I_num)
	overall_rating_tst, sps_tensor_useritemf_tst, useritem_ls_tst = \
			load_useritemfea('yelp_recursive_entry/yelp_recursive_test.entry', U_num, I_num)

	for key in sps_tensor_useritemf_trn.keys():
		if key[2] != F_num:
			sps_tensor_useritemf_trn[key] = 1 + 4/(1 + np.exp(0 - sps_tensor_useritemf_trn[key]))

	print('Preparing Data...')
	#load the interactions between user, item, feature and opinion word
	sps_tensor_userwordf_trn, sps_tensor_itemwordf_trn, feature_word_used = \
			load_useritemfeaword('yelp_recursive_entry/yelp_recursive_train.uifwords_entry',F_num,W_num)

	for key in sps_tensor_userwordf_trn.keys():
			sps_tensor_userwordf_trn[key] = 1 + 4 * ( 2 / (1 + np.exp(0 - sps_tensor_userwordf_trn[key]))-1)

	for key in sps_tensor_itemwordf_trn.keys():
			sps_tensor_itemwordf_trn[key] = 1 + 4 * ( 2 / (1 + np.exp(0 - sps_tensor_itemwordf_trn[key]))-1)
	
	os.makedirs('results/', exist_ok=True)
	fout_result = open('results/' + outfile + '.result', 'w')

	U_dim = args.useremb
	I_dim = args.itememb
	F_dim = args.featemb
	W_dim = args.wordemb
	lmd_BPR = args.bpr
	num_iter = args.iter

	num_processes = args.nprocess
	lr = args.lr

	print('Training started! Estimated time is being printed during training ...')
	(G1,G2,G3,U,I,F,W) = tsmtr.train(overall_rating_trn, 
					 sps_tensor_useritemf_trn, sps_tensor_userwordf_trn, sps_tensor_itemwordf_trn, 
					 useritem_ls_trn, U_dim, I_dim, F_dim, W_dim, U_num, I_num, F_num+1, W_num, 
					 lmd_BPR, num_iter, num_processes, lr=lr, 
					 cost_function='abs', random_seed=0, eps=1e-8)	

	params = {'G1': G1, 'G2': G2, 'G3': G3, 'U':U, 'I':I, 'F':F, 'W':W}
	with open('results/' + outfile + '.paras', 'wb') as output:
		pickle.dump(params, output) 

	fout_result.close()

