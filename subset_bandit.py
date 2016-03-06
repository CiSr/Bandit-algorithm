import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
user_movie=genfromtxt('x1base.dat',delimiter=',')

clf=PCA(n_components=10)
clf.fit(user_movie)
clf.explained_variance_ratio_
Top_Movies=clf.transform(user_movie)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(Top_Movies)

ratings_revised=X_train_minmax[0:10,0:6]

print ratings_revised

reward_list=[]

#Reward matrix for Items
temp=0
for j in range(0,6):
	temp=max(ratings_revised[:,j])
	reward_list.append(temp)
	print reward_list

nrange=6
#Choose the bandits randomly and update the q-value
#arms numbered from 1 to 5
arms = list(xrange(nrange))
maximum=0
bandit_sum=[]
epsilon=0.1

def banditcalculate():
	if(np.random.random() < epsilon):
		return max(reward_list)
	else:
		return random.choice(arms)
		
#The revised
a=banditcalculate()

#Update the bandits
counts=[0] * nrange
def update(arm):
	counts[arm]=counts[arm]+1
	no_times=counts[arm]
	value=reward_list[arm]
	new_value=((no_times - 1) / float(no_times))  + (1 / float(no_times)) -value
	reward_list[arm]=new_value
	return reward_list

empl=[0]*nrange
#empl=update(3)	

for i in range(0,6):
	temp=random.choice(arms)
	empl=update(temp)

print empl	