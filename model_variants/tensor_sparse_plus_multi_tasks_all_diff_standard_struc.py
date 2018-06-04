import autograd.numpy as np
from autograd import multigrad
import time
import random

cases = {
	1: {'HA': 'Ma, Nb -> MNab', 'HAT': 'MNab, Oab -> MNO', 'HA_sub': 'a, b -> ab', 'HAT_sub': 'ab, ab -> ','HAT_fea': 'ab,Oab -> O'},
	2: {'HA': 'Ma, Nab -> MNb', 'HAT': 'MNb, Ob -> MNO', 'HA_sub': 'a, ab -> b', 'HAT_sub': 'b, b -> ','HAT_fea': 'b,Ob -> O'},
	3: {'HA': 'Mab, Na -> MNb', 'HAT': 'MNb, Ob -> MNO', 'HA_sub': 'ab, a -> b', 'HAT_sub': 'b, b -> ','HAT_fea': 'b,Ob -> O'},
	4: {'HA': 'Ma, Na -> MNa', 'HAT': 'MNa, Oa -> MNO', 'HA_sub': 'a, a -> a', 'HAT_sub': 'a, a -> ','HAT_fea': 'a,Oa -> O'},
	5: {'H': 'abc, Ma -> Mbc', 'HA': 'Mbc, Nb -> MNc', 'HAT': 'MNc, Oc -> MNO', 'H_sub': 'abc, a -> bc', 'HA_sub': 'bc, b -> c', 'HAT_sub': 'c, c -> ', 'HA_fea': 'bc, b -> c', 'HAT_fea': 'c,Oc -> O'}
}

def multi_sps_case(H,A,T,case,index):
	tensor_value1 = np.einsum(cases[case]['HA_sub'], H[index[0]],A[index[1]])
	tensor_value2 = np.einsum(cases[case]['HAT_sub'], tensor_value1, T[index[2]])
	return tensor_value2

def multi_sps_feature_case(H,A,T,case,index):
	tensor_value1 = np.einsum(cases[case]['HA_sub'], H[index[0]],A[index[1]])
	tensor_value2 = np.einsum(cases[case]['HAT_fea'], tensor_value1, T)
	return tensor_value2
'''
#loss of item/feature recommendation task
	for kkk in range(element_num_iter):
		[key] = random.sample(element_list_useritemf,1)
		index = key[1:-1].split(',')
		for i in range(3):
			index[i] = int(index[i])

		U0I0 = np.einsum(cases[case]['HA_sub'],U0[index[0]],I0[index[1]])
		U1I1 = np.einsum(cases[case]['HA_sub'],U1[index[0]],I1[index[1]])
		tensor_value1 = np.einsum(cases[case]['HAT_sub'], U0I0,F0[index[2]]) 
		tensor_value2 = np.einsum(cases[case]['HAT_sub'], U1I1,F1[index[2]])
		tensor_value = tensor_value1 + tensor_value2
		error_square1 += (tensor_value - sps_tensor_useritemf[key])**2
	error_square1 = error_square1/element_num_iter
	
	BPR_pair_num_count = 0
	for sss in range(BPR_pair_num):
		[key] = random.sample(element_list_useritemf,1)
		index = key[1:-1].split(',')
		user = int(index[0])
		user_item_vector = overall_rating_matrix[user,:]
		item_i = int(index[1])
		item_j = int(I_num*random.random())

		if user_item_vector[item_i] < user_item_vector[item_j]:
			tmp = item_i
			item_i = item_j
			item_j = tmp
		if user_item_vector[item_i] == user_item_vector[item_j]:
			current_error2 = 0
		else:
			BPR_pair_num_count += 1
			U0I0 = np.einsum(cases[case]['HA_sub'],U0[user],I0[item_i]-I0[item_j])
			U1I1 = np.einsum(cases[case]['HA_sub'],U1[user],I1[item_i]-I1[item_j])
			tensor_value1 = np.einsum(cases[case]['HAT_sub'], U0I0,F0[F_num]) 
			tensor_value2 = np.einsum(cases[case]['HAT_sub'], U1I1,F1[F_num])
			over_diff = tensor_value1 + tensor_value2

			current_error2 = np.log(1/(1+np.exp(0 - over_diff)))
		error_BPR += current_error2
	error_BPR = error_BPR/BPR_pair_num_count

#loss of word recommendation task
	for kkk in range(element_num_iter):
		[key] = random.sample(element_list_userwordf,1)
		index = key[1:-1].split(',')
		#U = np.zeros(U0_dim + U1_dim)
		for i in range(3):
			index[i] = int(index[i])
		U0F0 = np.einsum(cases[case]['HA_sub'],U0[index[0]],F0[index[1]])
		U2F2 = np.einsum(cases[case]['HA_sub'],U2[index[0]],F2[index[1]])
		tensor_value1 = np.einsum(cases[case]['HAT_sub'], U0F0, W0[index[2]])
		tensor_value2 = np.einsum(cases[case]['HAT_sub'], U2F2, W1[index[2]])
		tensor_value = tensor_value1 + tensor_value2
		error_square2 += (tensor_value - sps_tensor_userwordf[key])**2
	error_square2 = error_square2/element_num_iter

	for kkk in range(element_num_iter):
		[key] = random.sample(element_list_itemwordf,1)
		index = key[1:-1].split(',')
		#I = np.zeros(I0_dim + I2_dim)
		for i in range(3):
			index[i] = int(index[i])

		I0F0 = np.einsum(cases[case]['HA_sub'],I0[index[0]],F0[index[1]])
		I2F3 = np.einsum(cases[case]['HA_sub'],I2[index[0]],F3[index[1]])
		tensor_value1 = np.einsum(cases[case]['HAT_sub'],I0F0,W0[index[2]])
		tensor_value2 = np.einsum(cases[case]['HAT_sub'],I2F3,W2[index[2]])
		tensor_value = tensor_value1 + tensor_value2 
		error_square3 += (tensor_value - sps_tensor_itemwordf[key])**2
	error_square3 = error_square3/element_num_iter
'''

def cost_abs_sparse_BPR_SGD(U0, U1, U2, I0, I1, I2, F0, F1, F2, F3, W0, W1, W2, sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, 
							element_list_useritemf, element_list_userwordf, element_list_itemwordf, overall_rating_matrix, 
							I_num, F_num, U0_dim, U1_dim, U2_dim, I0_dim, I1_dim, I2_dim, F0_dim, F1_dim, F2_dim, F3_dim, W0_dim,W1_dim,W2_dim, lmd_BPR, case):

	error_square1 = 0 
	error_square2 = 0
	error_square3 = 0
	error_BPR = 0
	element_num_iter = 50
	BPR_pair_num = 200
	lmd_reg = 0.05

	for kkk in range(element_num_iter):
		[key] = random.sample(element_list_useritemf,1)
		index = key[1:-1].split(',')
		for i in range(3):
			index[i] = int(index[i])

		tensor_value1 = np.einsum(cases[case]['HA_sub'], (U0+U1)[index[0]],(I0+I1)[index[1]])
		tensor_value = np.einsum(cases[case]['HAT_sub'], tensor_value1,(F0+F1)[index[2]])
		error_square1 += (tensor_value - sps_tensor_useritemf[key])**2
	error_square1 = error_square1/element_num_iter
	
	BPR_pair_num_count = 0
	for sss in range(BPR_pair_num):
		[key] = random.sample(element_list_useritemf,1)
		index = key[1:-1].split(',')
		user = int(index[0])
		user_item_vector = overall_rating_matrix[user,:]
		item_i = int(index[1])
		item_j = int(I_num*random.random())

		if user_item_vector[item_i] < user_item_vector[item_j]:
			tmp = item_i
			item_i = item_j
			item_j = tmp
		if user_item_vector[item_i] == user_item_vector[item_j]:
			current_error2 = 0
		else:
			BPR_pair_num_count += 1
			tensor_value1 = np.einsum(cases[case]['HA_sub'], (U0+U1)[user],(I0+I1)[item_i] - (I0+I1)[item_j])
			over_diff = np.einsum(cases[case]['HAT_sub'], tensor_value1,(F0+F1)[F_num])

			current_error2 = np.log(1/(1+np.exp(0 - over_diff)))
		error_BPR += current_error2
	error_BPR = error_BPR/BPR_pair_num_count

#loss of word recommendation task
	for kkk in range(element_num_iter):
		[key] = random.sample(element_list_userwordf,1)
		index = key[1:-1].split(',')
		#U = np.zeros(U0_dim + U1_dim)
		for i in range(3):
			index[i] = int(index[i])

		tensor_value1 = np.einsum(cases[case]['HA_sub'], (U0+U2)[index[0]],(F0+F2)[index[1]])
		tensor_value = np.einsum(cases[case]['HAT_sub'], tensor_value1,(W0+W1)[index[2]])

		error_square2 += (tensor_value - sps_tensor_userwordf[key])**2
	error_square2 = error_square2/element_num_iter

	for kkk in range(element_num_iter):
		[key] = random.sample(element_list_itemwordf,1)
		index = key[1:-1].split(',')
		#I = np.zeros(I0_dim + I2_dim)
		for i in range(3):
			index[i] = int(index[i])

		tensor_value1 = np.einsum(cases[case]['HA_sub'], (I0+I2)[index[0]],(F0+F3)[index[1]])
		tensor_value = np.einsum(cases[case]['HAT_sub'], tensor_value1,(W0+W2)[index[2]])
		error_square3 += (tensor_value - sps_tensor_itemwordf[key])**2
	error_square3 = error_square3/element_num_iter

	error_reg = 0
	error = U0.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*U1.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*U2.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = I0.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*I1.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*I2.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = F0.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*F1.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*F2.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*F3.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = W0.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*W1.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = 20000*W2.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	print('Least square1:' )
	print(error_square1)
	print('Least square2:' )
	print(error_square2)  
	print('Least square3:' )
	print(error_square3) 
	print('BPR:') 
	print(error_BPR)
	print("Total lost:")
	print(error_square1 + error_square2 + error_square3 - lmd_BPR*error_BPR)
	#return error1
	return error_square1 + error_square2 + error_square3 - lmd_BPR*error_BPR + lmd_reg*error_reg

def learn_HAT_SGD_adagrad(case, sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, U0_dim, U1_dim, U2_dim, I0_dim, I1_dim, I2_dim, F0_dim,F1_dim,F2_dim,F3_dim, W0_dim,W1_dim,W2_dim, U_num, I_num, F_num_1more, W_num, lmd_BPR,
								   num_iter=100000, lr=0.1, dis=False, cost_function='abs', U0_known=None, U1_known=None, 
								   U2_known=None, I0_known=None, I1_known=None, I2_known=None, F0_known=None, F1_known=None, F2_known=None, F3_known=None, W0_known=None, W1_known=None, W2_known=None, random_seed=0, eps=1e-8):

	F_num = F_num_1more - 1
	np.random.seed(random_seed)
	cost = cost_abs_sparse_BPR_SGD
	overall_rating_matrix = np.zeros((U_num,I_num))
	element_list_useritemf = list(sps_tensor_useritemf)
	element_list_userwordf = list(sps_tensor_userwordf)
	element_list_itemwordf = list(sps_tensor_itemwordf)

	for key in sps_tensor_useritemf:
		index = key[1:-1].split(',')
		for i in range(3):
			index[i] = int(index[i])
		if index[2] == F_num:
			overall_rating_matrix[int(index[0])][int(index[1])] = int(sps_tensor_useritemf[key])
	
	params = {}
	params['M'], params['N'], params['F'], params['W'] = (U_num,I_num,F_num,W_num) 

	print("users:" + str(params['M']))
	print("items:" + str(params['N']))
	print("features:" + str(params['F']))
	print("words:" + str(params['W']))

	U0_dim_initial = (U_num, U0_dim)
	U1_dim_initial = (U_num, U1_dim)
	U2_dim_initial = (U_num, U2_dim)
	I0_dim_initial = (I_num, I0_dim)
	I1_dim_initial = (I_num, I1_dim)
	I2_dim_initial = (I_num, I2_dim)
	F0_dim_initial = (F_num_1more, F0_dim)
	F1_dim_initial = (F_num_1more, F1_dim)
	F2_dim_initial = (F_num_1more, F2_dim)
	F3_dim_initial = (F_num_1more, F3_dim)
	W0_dim_initial = (W_num, W0_dim)
	W1_dim_initial = (W_num, W1_dim)
	W2_dim_initial = (W_num, W2_dim)

	U0 = np.random.rand(*U0_dim_initial)
	U1 = np.random.rand(*U1_dim_initial)
	U2 = np.random.rand(*U2_dim_initial)
	I0 = np.random.rand(*I0_dim_initial)
	I1 = np.random.rand(*I1_dim_initial)
	I2 = np.random.rand(*I2_dim_initial)
	F0 = np.random.rand(*F0_dim_initial)
	F1 = np.random.rand(*F1_dim_initial)
	F2 = np.random.rand(*F2_dim_initial)
	F3 = np.random.rand(*F3_dim_initial)
	W0 = np.random.rand(*W0_dim_initial)
	W1 = np.random.rand(*W1_dim_initial)
	W2 = np.random.rand(*W2_dim_initial)

	sum_square_gradients_U0 = np.zeros_like(U0)
	sum_square_gradients_U1 = np.zeros_like(U1)
	sum_square_gradients_U2 = np.zeros_like(U2)
	sum_square_gradients_I0 = np.zeros_like(I0)
	sum_square_gradients_I1 = np.zeros_like(I1)
	sum_square_gradients_I2 = np.zeros_like(I2)
	sum_square_gradients_F0 = np.zeros_like(F0)
	sum_square_gradients_F1 = np.zeros_like(F1)
	sum_square_gradients_F2 = np.zeros_like(F2)
	sum_square_gradients_F3 = np.zeros_like(F3)
	sum_square_gradients_W0 = np.zeros_like(W0)
	sum_square_gradients_W1 = np.zeros_like(W1)
	sum_square_gradients_W2 = np.zeros_like(W2)

	mg = multigrad(cost, argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

	# SGD procedure
	for i in range(num_iter):
		starttime = time.time()
		print(i+1)

		#print('?')
		del_u0, del_u1, del_u2, del_i0, del_i1, del_i2, del_f0, del_f1, del_f2, del_f3, del_w0, del_w1, del_w2 = mg(U0, U1, U2, I0, I1, I2, F0, F1, F2, F3, W0, W1, W2, 
												sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, 
												element_list_useritemf, element_list_userwordf, element_list_itemwordf, overall_rating_matrix, I_num, F_num, 
												U0_dim, U1_dim, U2_dim, I0_dim, I1_dim, I2_dim, F0_dim, F1_dim, F2_dim, F3_dim, W0_dim, W1_dim, W2_dim, lmd_BPR, case)

		sum_square_gradients_U0 += eps + np.square(del_u0)
		sum_square_gradients_U1 += eps + np.square(del_u1)
		sum_square_gradients_U2 += eps + np.square(del_u2)
		sum_square_gradients_I0 += eps + np.square(del_i0)
		sum_square_gradients_I1 += eps + np.square(del_i1)
		sum_square_gradients_I2 += eps + np.square(del_i2)
		sum_square_gradients_F0 += eps + np.square(del_f0)
		sum_square_gradients_F1 += eps + np.square(del_f1)
		sum_square_gradients_F2 += eps + np.square(del_f2)
		sum_square_gradients_F3 += eps + np.square(del_f3)
		sum_square_gradients_W0 += eps + np.square(del_w0)
		sum_square_gradients_W1 += eps + np.square(del_w1)
		sum_square_gradients_W2 += eps + np.square(del_w2)

		lr_u0 = np.divide(lr, np.sqrt(sum_square_gradients_U0))
		lr_u1 = np.divide(lr, np.sqrt(sum_square_gradients_U1))
		lr_u2 = np.divide(lr, np.sqrt(sum_square_gradients_U2))
		lr_i0 = np.divide(lr, np.sqrt(sum_square_gradients_I0))
		lr_i1 = np.divide(lr, np.sqrt(sum_square_gradients_I1))
		lr_i2 = np.divide(lr, np.sqrt(sum_square_gradients_I2))
		lr_f0 = np.divide(lr, np.sqrt(sum_square_gradients_F0))
		lr_f1 = np.divide(lr, np.sqrt(sum_square_gradients_F1))
		lr_f2 = np.divide(lr, np.sqrt(sum_square_gradients_F2))
		lr_f3 = np.divide(lr, np.sqrt(sum_square_gradients_F3))
		lr_w0 = np.divide(lr, np.sqrt(sum_square_gradients_W0))
		lr_w1 = np.divide(lr, np.sqrt(sum_square_gradients_W1))
		lr_w2 = np.divide(lr, np.sqrt(sum_square_gradients_W2))

		U0 -= lr_u0 * del_u0
		U1 -= lr_u1 * del_u1
		U2 -= lr_u2 * del_u2
		I0 -= lr_i0 * del_i0
		I1 -= lr_i1 * del_i1
		I2 -= lr_i2 * del_i2
		F0 -= lr_f0 * del_f0
		F1 -= lr_f1 * del_f1
		F2 -= lr_f2 * del_f2
		F3 -= lr_f3 * del_f3
		W0 -= lr_w0 * del_w0
		W1 -= lr_w1 * del_w1
		W2 -= lr_w2 * del_w2
		'''
		# Projection to known values
		if U0_known is not None:
			U0 = set_known(U0, U0_known)
		if U1_known is not None:
			U1 = set_known(U1, U1_known)
		if U2_known is not None:
			U2 = set_known(U2, U2_known)
		if I0_known is not None:
			I0 = set_known(I0, I0_known)
		if I1_known is not None:
			I1 = set_known(I1, I1_known)
		if I2_known is not None:
			I2 = set_known(I2, I2_known)
		if F0_known is not None:
			F0 = set_known(F0, F0_known)
		if F1_known is not None:
			F1 = set_known(F1, F1_known)
		if F2_known is not None:
			F2 = set_known(F2, F2_known)
		if F3_known is not None:
			F3 = set_known(F3, F3_known)
		if W0_known is not None:
			W0 = set_known(W0, W0_known)
		if W1_known is not None:
			W1 = set_known(W1, W1_known)
		if W2_known is not None:
			W2 = set_known(W2, W2_known)
		'''
		# Projection to non-negative space
		U0[U0 < 0] = 0
		U1[U1 < 0] = 0
		U2[U2 < 0] = 0
		I0[I0 < 0] = 0
		I1[I1 < 0] = 0
		I2[I2 < 0] = 0
		F0[F0 < 0] = 0
		F1[F1 < 0] = 0
		F2[F2 < 0] = 0
		F3[F3 < 0] = 0
		W0[W0 < 0] = 0
		W1[W1 < 0] = 0
		W2[W2 < 0] = 0
	
		nowtime = time.time()
		timeleft = (nowtime - starttime)*(num_iter-i-1) 

		if timeleft/60 > 60:
			print('time left: ' + str(int(timeleft/3600)) + ' hr ' + str(int(timeleft/60%60)) + ' min ' + str(int(timeleft%60)) + ' s')
		else:
			print("time left: " + str(int(timeleft/60)) + ' min ' + str(int(timeleft%60)) + ' s')

	return U0, U1, U2, I0, I1, I2, F0,F1,F2,F3, W0,W1,W2
