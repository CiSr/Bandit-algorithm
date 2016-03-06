import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
import random
user_movie=genfromtxt('x1base.dat',delimiter=',')

clf=PCA(n_components=10)
clf.fit(user_movie)
clf.explained_variance_ratio_
Top_Movies=clf.transform(user_movie)

[a,b]=Top_Movies.shape

for i in range(1,a):
	for j in range(1,b):
		Top_Movies[i,j]=abs(int(Top_Movies[i,j]))

print Top_Movies.shape

#Choose only 5 users	

Intermediate_array=np.empty((5,b))	

Intermediate_array=Top_Movies[0:5,:]


for i in range(0,5):
	for j in range(0,10):
		print abs(int(Intermediate_array[i,j]))

		
print Intermediate_array[:,0:b]		
sum1=0
initial_arm=random.randint(0,b)
sum1=sum1+Intermediate_array[:,initial_arm]


def choose_arm(self):

	if np.random.random()> epsilon:
		return np.argmax(self.values)
	else:
		return np.random.randint(n)	

reward_array=user_movie[0:5,1:10]
#print initial_array


#Finding the reward matrix
count=0
count_ones=0
[c,d]=reward_array.shape
for i in range(0,c):
	for j in range(0,d):
		if(reward_array[i,j]==0):
			count=count+1
			reward_array[i,j]=(reward_array[i,j]+1)/count
		else:
			count_ones=count_ones+1
			reward_array[i,j]=(reward_array[i,j])/(count+0.01)
print reward_array 


epsilon=0.1

arms=[]



