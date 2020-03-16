import numpy as np
from keras.layers import Dense, Input,Dropout,Embedding,LSTM,TimeDistributed,CuDNNLSTM
from keras.optimizers import SGD,Adam
from keras.models import Model
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
# pour l'utilisation de Raoh
configT = tf.ConfigProto()
configT.gpu_options.allow_growth = True
session = tf.Session(config=configT)

#réseau de neurones récurents 
#ou destructurer la séquence avec Sicitlearn (en utilisant un dictionnaire) et 

# fonction pour retourner la clé pour n'importe quelle valeur
def find_key(v,mots): 
	for k, val in mots.items(): 
		if v == val: 
			return k
	return -1
  

data=[]
prediction=[]

atis = open("atis.train","r")
lines=atis.readlines()
atis.close()
phrase=[]
mots={}
mots[0]=0
predphrase=[]
motspred={}
motspred[0]=0
comptmots=1
comptpred=1
for line in lines:
	line2 = line.split()
	if len(line2) > 0 :
		nmot=find_key(line2[0],mots)
		nmotPred=find_key(line2[1],motspred)
		if nmot == -1 :
			nmot=comptmots
			comptmots+=1
			mots[nmot]=line2[0]
		phrase.append(nmot)
		if nmotPred == -1:
			nmotPred=comptpred
			comptpred+=1
			motspred[nmotPred]=line2[1]
		predphrase.append(nmotPred)			
	else:
		while(len(phrase)!=86):
			phrase.append(0)
			predphrase.append(0)			
		data.append(phrase)
		prediction.append(predphrase)
		phrase=[]
		predphrase=[]
mots[comptmots]="inconnu"
#print(mots)
#print(motspred)
#print(data)
#print(prediction)
print(len(data))
print(len(prediction))

X=np.asarray(data)
Y=np.asarray(prediction)
Y=np.expand_dims(Y,2)
print(Y.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)
#print(len(X_train))
#print(len(X_test))

opti = SGD(lr=0.4, nesterov=False)#decay=1e−6, momentum=0.9
#opti = Adam()

#input: matrix (nbsamples∗inputsize)
nbsamples=len(data)#4978
MAX_SEQ_SIZE=86
hiddensize=1
tailleDictionnaire=len(mots)
nbLabels=len(motspred)
#X:(nbexamples∗MAX_SEQ_SIZE)
entree = Input(shape=(MAX_SEQ_SIZE,), dtype='int32')
emb = Embedding(tailleDictionnaire,100)(entree)
bi = CuDNNLSTM(15, return_sequences=True)(emb) #1er élem de LSTM : taille de la couche caché (mise arbitraire)
#bi = Bidirectional(LSTM(config.hidden, return_sequences=True))(emb)
drop = Dropout(0.5)(bi)
out = TimeDistributed(Dense(units=nbLabels,activation='softmax'))(drop)
#Y:(nbexamples∗MAX_SEQ_SIZE∗1)

model = Model(inputs=entree,outputs=out)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opti,metrics=['accuracy'])

#Y_train: output (labels vector, coded as numbers [0,nblabels])
model.fit(X_train, Y_train, epochs=20, batch_size=32)
predictions = model.predict(X_test).argmax(-1)
print(predictions)


atisRN = open("atisRN20prediction","w")
compt=0
print(len(predictions))
print(len(X_test))
print(len(Y_test))
for pred in predictions :
	if(compt==0):
		print("\npred=")
		print(pred)
		predreel=Y_test[compt] #prediction
		print("\npredreel=")
		print(predreel)
		phrase=X_test[compt] #data
		print("\nphrase=")
		print(phrase)
	compt+=1
	i=0
	while phrase[i]!=0:
		atisRN.write(str(mots[phrase[i]])+"\t"+str(motspred[predreel[i]]) + "\n")
		#t"+str(motspred[pred[i]])+"\n")
		print(str(mots[phrase[i]]) +"\t"+str(motspred[predreel[i]])+ "\n")
		#t"+ str(motspred[pred[i]])+"\n")
		i+=1
	atisRN.write("\n")
atisRN.close()
