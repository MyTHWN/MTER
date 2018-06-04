import random
import numpy as np

fin = open('/Users/nanwang/LabProject/virginia/AY_review_data/yelp_reviews/yelp_restaurant_recursive_entry/yelp_recursive.entry', encoding='UTF-8')
fout1 = open('/Users/nanwang/LabProject/virginia/AY_review_data/yelp_reviews/yelp_restaurant_recursive_entry/yelp_recursive_train.entry', 'w')
fout2 = open('/Users/nanwang/LabProject/virginia/AY_review_data/yelp_reviews/yelp_restaurant_recursive_entry/yelp_recursive_test.entry', 'w')
fout3 = open('/Users/nanwang/LabProject/virginia/AY_review_data/yelp_reviews/yelp_restaurant_recursive_entry/yelp_recursive_validate.entry', 'w')

M = 4303
N = 6534
F = 104

lines = fin.readlines()
train_ratio = 0.8
test_x = np.zeros((M,N))
list_length = 20

for line in lines:
	if (random.random() < train_ratio):
		fout1.writelines(line)
	else:
		fout2.writelines(line)
		eachline = line.strip().split(',')
		u_idx = int(eachline[0])
		i_idx = int(eachline[1])
		over_rating = int(eachline[2])
		if over_rating == 0:
			over_rating = 1
		test_x[u_idx][i_idx] = over_rating

for i in range(M):
	fout3.write('user:' + str(i) + '\n')
	print('user:' + str(i) )
	cnt = 0
	for j in range(N):
		if test_x[i][j] > 0 and cnt < list_length:
			fout3.write(str(j) + '\n')
			cnt += 1		
	while cnt<list_length:
		j = int((N-1)*random.random())
		if test_x[i][j] == 0:
			fout3.write(str(j) + '\n')
			cnt += 1
			test_x[i][j] += 1










