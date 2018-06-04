#Tensor Multi-task 9/11/2017 NanWang
import math
import random
import numpy as np
from sklearn.cluster import KMeans
import tensor_sparse_multi_tasks_all_diff as tsmtr

def get_index(key):
	index = key[1:-1].split(',')
	for i in range(3):
		index[i] = int(index[i])
	return index

fin_uif_train_entry = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/yelp_restaurant_recursive_entry_sigir/yelp_recursive_train.entry', encoding='UTF-8')
fin_uif_test_entry = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/yelp_restaurant_recursive_entry_sigir/yelp_recursive_test.entry', encoding='UTF-8')
#fin_uif_candidates = open('/Users/nanwang/LabProject/virginia/yelp_dataset_challenge_round9/recursive_modified_entry/yelp_recursive.rec_candidate', encoding='UTF-8')
fin_uifw_train_entry = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/yelp_restaurant_recursive_entry_sigir/yelp_recursive_train.uifwords_entry', encoding='UTF-8')
fin_uifw_test_entry = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/yelp_restaurant_recursive_entry_sigir/yelp_recursive_test.uifwords_entry', encoding='UTF-8')
#fin_uifw_candidates = open('/Users/nanwang/LabProject/virginia/yelp_dataset_challenge_round9/wordrec_modified_entry/yelp_recursive.uifwords_rec_candidate', encoding='UTF-8')
fin_feature_map = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/yelp_restaurant_recursive_entry_sigir/yelp_recursive.featuremap', encoding='UTF-8')
fin_word_map = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/yelp_restaurant_recursive_entry_sigir/yelp_recursive.wordmap', encoding='UTF-8')

#fout_itemrank = open('/net/zf8/nw6a/AY_Amazon_exp_sigir/Result/amazon_itemrec_15201012_forbpr800.reclist', 'w')
fout_featurerank = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/content_eval/yelp_featurerec_allshared.reclist', 'w')
fout_wordrank = open('/net/zf8/nw6a/AY_Yelp_exp_sigir/content_eval/yelp_wordrec_allshared.reclist', 'w')
#fout_rec_explanations = open('/net/zf8/nw6a/AY_Amazon_exp_sigir/Result/amazon_itemrec_15201012_forbpr800.explanations', 'w')

U0_dim = 15
U1_dim = 0
U2_dim = 0
I0_dim = 15
I1_dim = 0
I2_dim = 0
F0_dim = 10
F1_dim = 0
F2_dim = 0
F3_dim = 0
W0_dim = 12
W1_dim = 0
W2_dim = 0
lmd_BPR = 10

feature_maplines = fin_feature_map.readlines()
feature_map = {}


# create feature and words maping files
for line in feature_maplines:
	eachline = line.strip().split('=')
	feature_map[eachline[0]] = eachline[1]

word_maplines = fin_word_map.readlines()
word_map = {}

for line in word_maplines:
	eachline = line.strip().split('=')
	word_map[eachline[0]] = eachline[1]

U_num = 10719
I_num = 10410
F_num = 104
W_num = 1019

case = 5

sps_tensor_useritemf = {}
sps_tensor_useritemf_test = {}
sps_tensor_userwordf = {}
sps_tensor_itemwordf = {}
sps_tensor_useritemfw_test = {}
useritemfeature_test = {}
rec_expl_output = []
rec_word_output = []

overall_rating_train = np.zeros((U_num,I_num))
overall_rating_test = np.zeros((U_num,I_num))
rec_item = np.zeros((U_num,I_num))
item_feature_mentioned = np.zeros((I_num,F_num))
feature_word_used = np.zeros((F_num,W_num))
item_feature_num_record = np.zeros((I_num,1))

cnt_train = 0
cnt_test = 0

#read user-item-feature entries
uif_train_lines = fin_uif_train_entry.readlines()
uif_test_lines = fin_uif_test_entry.readlines()
for line in uif_train_lines:
	eachline = line.strip().split(',')
	ft_sent_pair = eachline[3].strip()
	if ft_sent_pair != '':
		u_idx = int(eachline[0])
		i_idx = int(eachline[1])
		over_rating = float(eachline[2])
		over_rating = int(over_rating)
		if over_rating == 0:
			over_rating = 1
		f_s_pairs = ft_sent_pair.strip().split(' ')
		for f_s in f_s_pairs:
			fea = f_s.strip().split(':')
			f_idx = int(fea[0])
			#user_feature_attention[u_idx][f_idx] += 1
			senti = int(fea[1])
			#item_feature_quality[i_idx][f_idx] += senti
			if str([u_idx,i_idx,f_idx]) not in sps_tensor_useritemf:
				sps_tensor_useritemf[str([u_idx,i_idx,f_idx])] = 0			
				cnt_train += 1
			sps_tensor_useritemf[str([u_idx,i_idx,f_idx])] += senti 
			if item_feature_mentioned[i_idx][f_idx] == 0:
				item_feature_num_record[i_idx] += 1
			item_feature_mentioned[i_idx][f_idx] += 1
		overall_rating_train[u_idx,i_idx] = over_rating
		sps_tensor_useritemf[str([u_idx,i_idx,F_num])] = over_rating

for line in uif_test_lines:
	eachline = line.strip().split(',')
	ft_sent_pair = eachline[3].strip()
	if ft_sent_pair != '':
		u_idx = int(eachline[0])
		i_idx = int(eachline[1])
		over_rating = float(eachline[2])
		over_rating = int(over_rating)
		if over_rating == 0:
			over_rating = 1
		f_s_pairs = ft_sent_pair.strip().split(' ')
		for f_s in f_s_pairs:
			fea = f_s.strip().split(':')
			f_idx = int(fea[0])
			senti = int(fea[1])
			if str([u_idx,i_idx,f_idx]) not in sps_tensor_useritemf_test:
				sps_tensor_useritemf_test[str([u_idx,i_idx,f_idx])] = 0
				cnt_test += 1
			sps_tensor_useritemf_test[str([u_idx,i_idx,f_idx])] += senti
			if item_feature_mentioned[i_idx][f_idx] == 0:
				item_feature_num_record[i_idx] += 1
			item_feature_mentioned[i_idx][f_idx] += 1
		overall_rating_test[u_idx,i_idx] = over_rating
		sps_tensor_useritemf_test[str([u_idx,i_idx,F_num])] = over_rating
'''
for key in sps_tensor_useritemf.keys():
	index = get_index(key)
	if index[2] != F_num:
		sps_tensor_useritemf[key] = 1 + 4/(1 + np.exp(0 - sps_tensor_useritemf[key]))

for key in sps_tensor_useritemf_test.keys():
	index = get_index(key)
	if index[2] != F_num:
		sps_tensor_useritemf_test[key] = 1 + 4/(1 + np.exp(0 - sps_tensor_useritemf_test[key]))
'''
#read user/item-feture-word entries
uifw_train_lines = fin_uifw_train_entry.readlines()
uifw_test_lines = fin_uifw_test_entry.readlines()
for line in uifw_train_lines:
	eachline = line.strip().split(',')
	u_idx = int(eachline[0])
	i_idx = int(eachline[1])
	f_idx = int(eachline[2])
	w_idx = int(eachline[3])

	feature_word_used[f_idx][w_idx] += 1

	if str([u_idx,f_idx,w_idx]) not in sps_tensor_userwordf:
		sps_tensor_userwordf[str([u_idx,f_idx,w_idx])] = 0
	sps_tensor_userwordf[str([u_idx,f_idx,w_idx])] += 1

	if str([i_idx,f_idx,w_idx]) not in sps_tensor_itemwordf:
		sps_tensor_itemwordf[str([i_idx,f_idx,w_idx])] = 0
	sps_tensor_itemwordf[str([i_idx,f_idx,w_idx])] += 1
'''
for key in sps_tensor_userwordf.keys():
	sps_tensor_userwordf[key] = 1 + 4/(1 + np.exp(0 - sps_tensor_userwordf[key]))
for key in sps_tensor_itemwordf.keys():
	sps_tensor_itemwordf[key] = 1 + 4/(1 + np.exp(0 - sps_tensor_itemwordf[key]))
'''
for line in uifw_test_lines:
	cnt_train += 1
	eachline = line.strip().split(',')
	u_idx = int(eachline[0])
	i_idx = int(eachline[1])
	f_idx = int(eachline[2])
	w_idx = int(eachline[3])
	#feature_word_used[f_idx][w_idx] += 1

	if str([u_idx,i_idx,f_idx,w_idx]) not in sps_tensor_useritemfw_test:
		sps_tensor_useritemfw_test[str([u_idx,i_idx,f_idx,w_idx])] = 0
	sps_tensor_useritemfw_test[str([u_idx,i_idx,f_idx,w_idx])] += 1
	useritemfeature_test[(u_idx,i_idx,f_idx)] = 1

print("train size:" + str(cnt_train) + '\n')
print("test size:" + str(cnt_test) + '\n')



#training
(G1,G2,G3,U0,U1,U2,I0,I1,I2,F0,F1,F2,F3,W0,W1,W2) = tsmtr.learn_HAT_SGD_adagrad(case, sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, 
								   U0_dim, U1_dim, U2_dim, I0_dim, I1_dim, I2_dim, F0_dim,F1_dim,F2_dim,F3_dim, W0_dim,W1_dim,W2_dim, U_num, I_num, F_num+1, W_num, lmd_BPR,
								   num_iter=150000, lr=0.1, dis=False, cost_function='abs', G1_known=None, G2_known=None, G3_known = None, U0_known=None, U1_known=None, 
								   U2_known=None, I0_known=None, I1_known=None, I2_known=None, F0_known=None, F1_known=None, F2_known=None, F3_known=None, W0_known=None, W1_known=None, W2_known=None, random_seed=0, eps=1e-8)

U_task1 = np.concatenate((U0,U1),axis=1)
U_task2 = np.concatenate((U0,U2),axis=1)
I_task1 = np.concatenate((I0,I1),axis=1)
I_task2 = np.concatenate((I0,I2),axis=1)
F_task1 = np.concatenate((F0,F1),axis=1)
F_task2_u = np.concatenate((F0,F2),axis=1)
F_task2_i = np.concatenate((F0,F3),axis=1)
W_task2_u = np.concatenate((W0,W1),axis=1)
W_task2_i = np.concatenate((W0,W2),axis=1)

#generating item recommendation lists
print('Generating item recommendation ranking lists and explanations ...')
 

list_length = 100
top_feature_num = 5
rec_item_num = 0
temp_feature_vector = []

for i in range(U_num):
	print(i)
	rec_item_num = 0
	itemrec_for_user = np.zeros((I_num,1))
	purchased = 0
	for jj in range(I_num) :
		if overall_rating_train[i][jj] > 0:
			itemrec_for_user[jj] = 0
		else:
			itemrec_for_user[jj] = rec_item[i,jj]
	temp = {}
	temp_list = []
	top_item = 0

	for iii in range(I_num):
		if overall_rating_test[i][iii] > 0:
			purchased = 1
			temp_list.append(iii)
	if purchased == 1:
		for key in temp_list:
			temp_feature_vector = []
			if overall_rating_test[i][int(key)] > 0:
				fout_featurerank.write('###Item:' + str(key) + ' reclist: \n')
				temp_feature_vector = (tsmtr.multi_sps_feature_case(G1,U_task1,I_task1,F_task1,case,[i,int(key)]))[0:F_num]
				for fff in range(F_num):
					if item_feature_mentioned[int(key)][fff] == 0:
						temp_feature_vector[fff] = 0
				for j in range(F_num):
					lct_top_feature = np.where(temp_feature_vector == np.max(temp_feature_vector))[0][0]
					if str([i,int(key),lct_top_feature]) in sps_tensor_useritemf_test:
						fout_featurerank.write('$$$mentioned feature: %-8d'%(lct_top_feature) + ' Real rating: %-8d'%(sps_tensor_useritemf_test[str([i,int(key),lct_top_feature])])+ '\n')
						fout_wordrank.write('###Feature: %-8d'%(lct_top_feature) + ' reclist: \n')
						temp_word_vector = tsmtr.multi_sps_feature_case(G2,U_task2,F_task2_u[0:F_num],W_task2_u,case,[i,int(lct_top_feature)])
						item_word_vector = tsmtr.multi_sps_feature_case(G3,I_task2,F_task2_i[0:F_num],W_task2_i,case,[int(key),int(lct_top_feature)])
						for www in range(W_num):
							if feature_word_used[lct_top_feature][www] == 0:
								temp_word_vector[www] = 0
							temp_word_vector[www] *= item_word_vector[www]
						for www in range(100):
							lct_top_word = np.where(temp_word_vector == np.max(temp_word_vector))[0][0]
							if str([i,int(key),lct_top_feature,lct_top_word]) in sps_tensor_useritemfw_test:
								fout_wordrank.write('$$$Used word: %-8d'%(lct_top_word) + ' Real rating: %-8d'%(sps_tensor_useritemfw_test[str([i,int(key),lct_top_feature,lct_top_word])])+ '\n')
							else:
								fout_wordrank.write('$$$Word: %-8d'%(lct_top_word) + ' Real rating: %-8d'%(0)+ '\n')
							temp_word_vector[lct_top_word] = -1
					else:
						fout_featurerank.write('feature: %-8d'%(lct_top_feature) + ' Real rating: %-8d'%(0)+ '\n')
					temp_feature_vector[lct_top_feature] = -1








