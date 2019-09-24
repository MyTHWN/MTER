import autograd.numpy as np
from autograd import grad
import time
import random
import multiprocessing as mp
import ctypes as c

def get_value(G,U,I,F,index):
	tensor_value1 = np.einsum('abc, a -> bc', G, U[index[0]])
	tensor_value2 = np.einsum('bc, b -> c', tensor_value1, I[index[1]])
	return np.einsum('c, c -> ', tensor_value2, F[index[2]])


def sign(a,b):
	return 1 if a > b else -1

def rmse(G,U,I,F,sps_tensor_useritemf):
	sqerror = 0
	for key in sps_tensor_useritemf.keys():
		pred_rating = get_value(G,U,I,F,key)
		#print(pred_rating)
		sqerror += (pred_rating - sps_tensor_useritemf[key]) ** 2

	return np.sqrt(sqerror/len(sps_tensor_useritemf.keys()))

def squrerror(G,U,I,F,key,true_rating):
	pred_rating = get_value(G,U,I,F,key)
	return (pred_rating-true_rating)**2


def grad_worker_mse(sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, lmd_BPR, G1, G2, G3, U, I, F, W, \
					error_square, error_bpr, lock, q_samples_mse, \
                   del_g1, del_g2, del_g3, del_u, del_i, del_f, del_w, num_grad):
	eps = 1e-8
	while 1:
		if not q_samples_mse.empty():
			sample = q_samples_mse.get()
			if not sample:
				break

			mse_sample = sample

			for key in mse_sample[0]:
				#print(key)
				pred_rating = get_value(G1, U, I, F, key)
				#print(pred_rating, sps_tensor_useritemf[key])
				del_sqerror = 2 * (pred_rating - sps_tensor_useritemf[key])
				lock.acquire()
				error_square.value += (pred_rating - sps_tensor_useritemf[key])**2
				del_g1 += del_sqerror * np.einsum('ab,c->abc', np.einsum('a,b -> ab', U[key[0]], I[key[1]]), F[key[2]])
				del_u[key[0]] += del_sqerror * np.einsum('ac,c->a', np.einsum('abc,b->ac',G1,I[key[1]]), F[key[2]])
				del_i[key[1]] += del_sqerror * np.einsum('bc,c->b', np.einsum('abc,a->bc',G1,U[key[0]]), F[key[2]])
				del_f[key[2]] += del_sqerror * np.einsum('bc,b->c', np.einsum('abc,a->bc',G1,U[key[0]]), I[key[1]])
				lock.release()

			for key in mse_sample[1]:
				pred_rating = get_value(G2, U, F, W, key)
				#print(pred_rating, sps_tensor_userwordf[key])
				del_sqerror = 2 * (pred_rating - sps_tensor_userwordf[key])
				lock.acquire()
				error_square.value += (pred_rating - sps_tensor_userwordf[key])**2
				del_g2 += del_sqerror * np.einsum('ab,c->abc', np.einsum('a,b -> ab', U[key[0]], F[key[1]]), W[key[2]])
				del_u[key[0]] += del_sqerror * np.einsum('ac,c->a', np.einsum('abc,b->ac',G2,F[key[1]]), W[key[2]])
				del_f[key[1]] += del_sqerror * np.einsum('bc,c->b', np.einsum('abc,a->bc',G2,U[key[0]]), W[key[2]])
				del_w[key[2]] += del_sqerror * np.einsum('bc,b->c', np.einsum('abc,a->bc',G2,U[key[0]]), F[key[1]])
				lock.release()

			for key in mse_sample[2]:
				pred_rating = get_value(G3, I, F, W, key)
				#print(pred_rating, sps_tensor_itemwordf[key])
				del_sqerror = 2 * (pred_rating - sps_tensor_itemwordf[key])
				lock.acquire()
				error_square.value += (pred_rating - sps_tensor_itemwordf[key])**2
				del_g3 += del_sqerror * np.einsum('ab,c->abc', np.einsum('a,b -> ab', I[key[0]], F[key[1]]), W[key[2]])
				del_i[key[0]] += del_sqerror * np.einsum('ac,c->a', np.einsum('abc,b->ac',G3,F[key[1]]), W[key[2]])
				del_f[key[1]] += del_sqerror * np.einsum('bc,c->b', np.einsum('abc,a->bc',G3,I[key[0]]), W[key[2]])
				del_w[key[2]] += del_sqerror * np.einsum('bc,b->c', np.einsum('abc,a->bc',G3,I[key[0]]), F[key[1]])
				lock.release()

			lock.acquire()
			num_grad.value += 1
			lock.release()


def grad_worker_bpr(sps_tensor_useritemf, overall_rating_matrix, lmd_BPR, G1, U, I, F, error_square, error_bpr, lock, q_samples_bpr, \
                   del_g1, del_u, del_i, del_f, num_grad):
	eps = 1e-8
	while 1:
		if not q_samples_bpr.empty():
			sample = q_samples_bpr.get()
			if not sample:
				break

			bpr_sample_ele = sample[0]
			item2_sample = sample[1]
			
			for i,key in enumerate(bpr_sample_ele):
				user = key[0]
				item_i = key[1]
				item_j = item2_sample[i]
				user_item_vector = overall_rating_matrix[user,:]

				if user_item_vector[item_i] != user_item_vector[item_j]:
					pred_x_ij = (get_value(G1, U, I, F, (user,item_i,-1))-get_value(G1, U, I, F, (user,item_j,-1)))*sign(user_item_vector[item_i],user_item_vector[item_j])
					del_bpr = lmd_BPR * (np.exp(-pred_x_ij) / (1+np.exp(-pred_x_ij))) * sign(user_item_vector[item_i],user_item_vector[item_j])
					
					lock.acquire()
					error_bpr.value += np.log(1/(1+np.exp(-pred_x_ij)))
					del_g1 -= del_bpr * np.einsum('ab,c->abc', np.einsum('a,b -> ab', U[key[0]], I[item_i] - I[item_j]),F[-1])
					del_u[user] -= del_bpr * np.einsum('ac,c->a', np.einsum('abc,b->ac',G1,I[item_i]-I[item_j]), F[-1])
					del_i[item_i] -= del_bpr * np.einsum('bc,c->b', np.einsum('abc,a->bc',G1,U[key[0]]), F[-1])
					del_i[item_j] += del_bpr * np.einsum('bc,c->b', np.einsum('abc,a->bc',G1,U[key[0]]), F[-1])
					del_f[-1] -= del_bpr * np.einsum('bc,b->c', np.einsum('abc,a->bc',G1,U[key[0]]), I[item_i]-I[item_j])
					lock.release()

			lock.acquire()
			num_grad.value += 1
			lock.release()
			

def paraserver(useritem_ls, sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, \
				   lmd_BPR, num_iter, lr, G1, G2, G3, U, I, F, W, error_square, error_bpr, q_samples_mse, q_samples_bpr, \
                   del_g1, del_g2, del_g3, del_u, del_i, del_f, del_w, num_grad, num_processes):
	eps = 1e-6
	print('Training Started')

	element_num_iter = 50
	BPR_pair_num = 1000
	lmd_reg = 0.1

	sum_square_gradients_G1 = np.zeros_like(G1)
	sum_square_gradients_G2 = np.zeros_like(G2)
	sum_square_gradients_G3 = np.zeros_like(G3)
	sum_square_gradients_U = np.zeros_like(U)
	sum_square_gradients_I = np.zeros_like(I)
	sum_square_gradients_F = np.zeros_like(F)
	sum_square_gradients_W = np.zeros_like(W)

	mse_per_proc = int(element_num_iter/num_processes)
	bpr_per_proc = int(BPR_pair_num/num_processes)

	for iteration in range(num_iter):
		starttime = time.time()
		print('iteration:', iteration+1, '/', num_iter)

		error_square.value = 0
		error_bpr.value = 0

		mse_sample_1 = random.sample(sps_tensor_useritemf.keys(), element_num_iter)
		mse_sample_2 = random.sample(sps_tensor_userwordf.keys(), element_num_iter)
		mse_sample_3 = random.sample(sps_tensor_itemwordf.keys(), element_num_iter)
		bpr_sample_ele = random.sample(useritem_ls, BPR_pair_num)
		item2_sample = random.sample(range(0, I.shape[0]), BPR_pair_num)
        
		num_grad.value = 0
		del_g1[:] = 0
		del_g2[:] = 0
		del_g3[:] = 0
		del_u[:] = 0 
		del_i[:] = 0 
		del_f[:] = 0
		del_w[:] = 0

		for i in range(num_processes):
			q_samples_mse.put((mse_sample_1[mse_per_proc*i:mse_per_proc*(i+1)], mse_sample_2[mse_per_proc*i:mse_per_proc*(i+1)], mse_sample_3[mse_per_proc*i:mse_per_proc*(i+1)]))
			q_samples_bpr.put((bpr_sample_ele[bpr_per_proc*i:bpr_per_proc*(i+1)], item2_sample[bpr_per_proc*i:bpr_per_proc*(i+1)]))

		while 1:
			if num_grad.value == 2 * num_processes:
				break

		del_g1_reg = del_g1 + lmd_reg * G1 * (del_g1 != 0)
		del_g2_reg = del_g2 + lmd_reg * G2 * (del_g2 != 0)
		del_g3_reg = del_g3 + lmd_reg * G3 * (del_g3 != 0)
		del_u_reg = del_u + lmd_reg * U * (del_u != 0)
		del_i_reg = del_i + lmd_reg * I * (del_i != 0)
		del_f_reg = del_f + lmd_reg * F * (del_f != 0)
		del_w_reg = del_w + lmd_reg * W * (del_w != 0)
		
		sum_square_gradients_G1 += eps + np.square(del_g1_reg)
		sum_square_gradients_G2 += eps + np.square(del_g2_reg)
		sum_square_gradients_G3 += eps + np.square(del_g3_reg)
		sum_square_gradients_U += eps + np.square(del_u_reg)
		sum_square_gradients_I += eps + np.square(del_i_reg)
		sum_square_gradients_F += eps + np.square(del_f_reg)
		sum_square_gradients_W += eps + np.square(del_w_reg)

		lr_g1 = np.divide(lr, np.sqrt(sum_square_gradients_G1))
		lr_g2 = np.divide(lr, np.sqrt(sum_square_gradients_G2))
		lr_g3 = np.divide(lr, np.sqrt(sum_square_gradients_G3))
		lr_u = np.divide(lr, np.sqrt(sum_square_gradients_U))
		lr_i = np.divide(lr, np.sqrt(sum_square_gradients_I))
		lr_f = np.divide(lr, np.sqrt(sum_square_gradients_F))
		lr_w = np.divide(lr, np.sqrt(sum_square_gradients_W))

		G1 -= lr_g1 * del_g1_reg
		G2 -= lr_g2 * del_g2_reg
		G3 -= lr_g3 * del_g3_reg
		U -= lr_u * del_u_reg
		I -= lr_i * del_i_reg
		F -= lr_f * del_f_reg
		W -= lr_w * del_w_reg

		# Projection to non-negative space
		G1[G1 < 0] = 0
		G2[G2 < 0] = 0
		G3[G3 < 0] = 0
		U[U < 0] = 0
		I[I < 0] = 0
		F[F < 0] = 0
		W[W < 0] = 0

		if element_num_iter: print('RMSE:', np.sqrt(error_square.value / 3 / element_num_iter))
		if BPR_pair_num: print('BPR:', error_bpr.value / BPR_pair_num)
		print('------------------------------------------------------------------------------')

		nowtime = time.time()
		timeleft = (nowtime - starttime)*(num_iter-iteration-1) 

		if timeleft/60 > 60:
			print('time left: ' + str(int(timeleft/3600)) + ' hr ' + str(int(timeleft/60%60)) + ' min ' + str(int(timeleft%60)) + ' s')
		else:
			print("time left: " + str(int(timeleft/60)) + ' min ' + str(int(timeleft%60)) + ' s')
    
	for _ in range(num_processes):
		q_samples_bpr.put(0)
		q_samples_mse.put(0)


def train(overall_rating_matrix, sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, useritem_ls, \
			 U_dim, I_dim, F_dim, W_dim, U_num, I_num, F_num, W_num, lmd_BPR, \
			 num_iter, num_processes, lr, cost_function='abs', random_seed=0, eps=1e-8):
		
	np.random.seed(random_seed)

	U_dim_initial = (U_num, U_dim)
	I_dim_initial = (I_num, I_dim)
	F_dim_initial = (F_num, F_dim)
	W_dim_initial = (W_num, W_dim)
	G1_dim_initial = (U_dim, I_dim, F_dim)
	G2_dim_initial = (U_dim, F_dim, W_dim)
	G3_dim_initial = (I_dim, F_dim, W_dim)

	mp_U = mp.Array(c.c_double, np.random.rand(U_num * U_dim))  # shared among multiple processes
	mp_I = mp.Array(c.c_double, np.random.rand(I_num * I_dim))
	mp_F = mp.Array(c.c_double, np.random.rand(F_num * F_dim))
	mp_W = mp.Array(c.c_double, np.random.rand(W_num * W_dim))
	mp_G1 = mp.Array(c.c_double, np.random.rand(U_dim * I_dim * F_dim))
	mp_G2 = mp.Array(c.c_double, np.random.rand(U_dim * F_dim * W_dim))
	mp_G3 = mp.Array(c.c_double, np.random.rand(I_dim * F_dim * W_dim))

	G1 = np.frombuffer(mp_G1.get_obj()).reshape(G1_dim_initial)
	G2 = np.frombuffer(mp_G2.get_obj()).reshape(G2_dim_initial)
	G3 = np.frombuffer(mp_G3.get_obj()).reshape(G3_dim_initial)
	U = np.frombuffer(mp_U.get_obj()).reshape(U_dim_initial)  # point to the same shared memory
	I = np.frombuffer(mp_I.get_obj()).reshape(I_dim_initial)
	F = np.frombuffer(mp_F.get_obj()).reshape(F_dim_initial)
	W = np.frombuffer(mp_W.get_obj()).reshape(W_dim_initial)

	mp_del_g1_arr = mp.Array(c.c_double, U_dim * I_dim * F_dim) # shared, used from multiple processes
	mp_del_g2_arr = mp.Array(c.c_double, U_dim * F_dim * W_dim)
	mp_del_g3_arr = mp.Array(c.c_double, I_dim * F_dim * W_dim)
	mp_del_u_arr = mp.Array(c.c_double, U_num * U_dim) 
	mp_del_i_arr = mp.Array(c.c_double, I_num * I_dim) 
	mp_del_f_arr = mp.Array(c.c_double, F_num * F_dim) 
	mp_del_w_arr = mp.Array(c.c_double, W_num * W_dim) 

	del_g1 = np.frombuffer(mp_del_g1_arr.get_obj()).reshape(G1_dim_initial)
	del_g2 = np.frombuffer(mp_del_g2_arr.get_obj()).reshape(G2_dim_initial)
	del_g3 = np.frombuffer(mp_del_g3_arr.get_obj()).reshape(G3_dim_initial)
	del_u = np.frombuffer(mp_del_u_arr.get_obj()).reshape(U_dim_initial)
	del_i = np.frombuffer(mp_del_i_arr.get_obj()).reshape(I_dim_initial)
	del_f = np.frombuffer(mp_del_f_arr.get_obj()).reshape(F_dim_initial)
	del_w = np.frombuffer(mp_del_w_arr.get_obj()).reshape(W_dim_initial)

	lock = mp.Lock()
	q_samples_mse = mp.Queue()
	q_samples_bpr = mp.Queue()

	num_grad = mp.Value('i', 0) 
	error_square = mp.Value('d', 0)
	error_bpr = mp.Value('d', 0)

	processes = []
	ps = mp.Process(target=paraserver, \
                       args=(useritem_ls, sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, \
				   			lmd_BPR, num_iter, lr, G1, G2, G3, U, I, F, W, error_square, error_bpr, q_samples_mse, q_samples_bpr, \
                   			del_g1, del_g2, del_g3, del_u, del_i, del_f, del_w, num_grad, num_processes))

	ps.start()
	processes.append(ps)

    #processes for U, V, W
	for _ in range(num_processes):
		p = mp.Process(target=grad_worker_mse, \
                       args=(sps_tensor_useritemf, sps_tensor_userwordf, sps_tensor_itemwordf, lmd_BPR, G1, G2, G3, U, I, F, W, \
                       		error_square, error_bpr, lock, q_samples_mse, \
                   			del_g1, del_g2, del_g3, del_u, del_i, del_f, del_w, num_grad))
		processes.append(p)
		p.start()
	
	for _ in range(num_processes):
		p = mp.Process(target=grad_worker_bpr, \
                       args=(sps_tensor_useritemf, overall_rating_matrix, lmd_BPR, G1, U, I, F, error_square, error_bpr, lock, q_samples_bpr, \
                   			del_g1, del_u, del_i, del_f, num_grad))
		processes.append(p)
		p.start()

	for process in processes:
		process.join()

	return G1, G2, G3, U, I, F, W



