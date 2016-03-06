import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
from  matplotlib import pyplot as plt

user_movie=genfromtxt('x1base.dat',delimiter=',')

clf=PCA(n_components=100)
clf.fit(user_movie)
clf.explained_variance_ratio_
Top_Movies=clf.transform(user_movie)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(Top_Movies)

ratings_revised=X_train_minmax[0:100,0:50]


#plt.show()
print ratings_revised
[c,d]=ratings_revised.shape

print (c,d)
Reward=np.empty((c,d))
Reward=ratings_revised



[nrows, ncols]=Reward.shape
print nrows
reward=[]
arms = list(xrange(nrows))
print arms
bandits=[]
temp=0
for i in range(0,ncols):
	select=random.choice(arms)
	temp=max(Reward[select,:])
	bandits.append(temp)
	print "Bandit values"
	print bandits

#epsilon=0.1


def epsilon_update():
	total = np.sum(bandits)
	print "Total bandits used:"
	print total

def choose_arm():
	epsilon=epsilon_update()
	if(np.random.random() > epsilon):
		return np.argmax(bandits)	
	else:
		return np.random.randint(ncols)



counts=[0]*ncols
delta_distance=0.01

def update(arm):
	temp=counts[arm] + 1
	counts.append(temp)
	n=(counts[arm])+0.12
	value = bandits[arm]
# Running product
	new_value = ((n - 1) / float(n)) * value + (1 / float(n)) 
	bandits[arm] = abs((new_value+delta_distance)/(random.uniform(0,1)))
	

for i in range(1,100):
	a=choose_arm()
	update(a)

print bandits
prebands=[]

for i in range(0,50):
	temp_value=random.uniform(0,1)
	prebands.append(temp_value)
#plt.plot(bandits);plt.show()

#print ratings_revised[1,:]*bandits


print prebands

temp=np.empty((nrows,ncols))


for i in range(0,12):
	temp=abs(((ratings_revised[i,]*prebands*10)-1.0))
	print temp

