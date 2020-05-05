import numpy as np
from keras.layers import Dense, Input,Dropout,Embedding,LSTM,TimeDistributed,CuDNNLSTM
from keras.optimizers import SGD,Adam
from keras.models import Model
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split

if sys.argv[1]=="-raoh":
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

opti = SGD(lr=0.4, nesterov=False,decay=0.000001, momentum=0.9)
#opti = Adam()

#input: matrix (nbsamples∗inputsize)
nbsamples=len(data)#4978
MAX_SEQ_SIZE=86
hiddensize=2
tailleDictionnaire=len(mots)
nbLabels=len(motspred)
#X:(nbexamples∗MAX_SEQ_SIZE)
entree = Input(shape=(MAX_SEQ_SIZE,), dtype='int32')
emb = Embedding(tailleDictionnaire,100)(entree)
if sys.argv[1]=="-raoh":
	bi = CuDNNLSTM(15, return_sequences=True)(emb)
else:
	bi = LSTM(15, return_sequences=True)(emb) #1er élem de LSTM : taille de la couche caché (mise arbitraire) ou CuDNNLSTM sur Raoh pour que ça aille plus vite
drop = Dropout(0.7)(bi)
out = TimeDistributed(Dense(units=nbLabels,activation='softmax'))(drop)
#Y:(nbexamples∗MAX_SEQ_SIZE∗1)

model = Model(inputs=entree,outputs=out)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opti,metrics=['accuracy'])
nbEpoques=50
#Y_train: output (labels vector, coded as numbers [0,nblabels])
model.fit(X_train, Y_train, epochs=nbEpoques, batch_size=32)
predictions = model.predict(X_test).argmax(-1)
print(predictions)


atisRN = open("atisRN"+str(nbEpoques)+"prediction","w")
compt=0
print(len(predictions))
print(len(X_test))
print(len(Y_test))
for pred in predictions :
	predreel=Y_test[compt] #prediction
	phrase=X_test[compt] #data
	if(compt==0):
		print("\npred=")
		print(pred)
		print("\npredreel=")
		print(predreel)
		print("\nphrase=")
		print(phrase)
	compt+=1
	i=0
	while phrase[i]!=0:
		mot=mots[phrase[i]]
		predmotreel=motspred[predreel[i][0]]	
		predmot=motspred[pred[i]]
		atisRN.write(mot)
		atisRN.write("\t")
		atisRN.write(str(predmotreel))
		atisRN.write("\t")
		atisRN.write(str(predmot))
		atisRN.write("\n")
		i+=1
	atisRN.write("\n")
atisRN.close()

if False:
	atisEval = open("atis.test","r")
	linesEval=atisEval.readlines()
	atisEval.close()
	phraseEval=[]
	eval=[]
	for line in linesEval:
		line2 = line.split()
		if len(line2) > 0 :
			nmot=find_key(line2[0],mots)
			if nmot == -1 :
				nmot=comptmots
				comptmots+=1
				mots[nmot]=line2[0]
			phraseEval.append(nmot)
		else:
			while(len(phraseEval)!=86):
				phraseEval.append(0)
			eval.append(phraseEval)
			phraseEval=[]
	XEval=np.asarray(eval)
	predictionsEval = model.predict(XEval).argmax(-1)
	print(predictionsEval)

	atisRun = open("Julio_SANTILARIO-BERTHILIER_Augustin_JANVIER_system2(réseau de neurones)-run3","w")
	compt=0
	print(len(predictions))
	for pred in predictions :
		compt+=1
		phrase=XEval[compt] #data
		i=0
		while phrase[i]!=0:
			mot=mots[phrase[i]]
			predmot=motspred[pred[i]]
			atisRun.write(mot)
			atisRun.write("\t")
			atisRun.write(str(predmot))
			atisRun.write("\n")
			i+=1
		atisRun.write("\n")
	atisRun.close()

#Test sur une partie du train à ~ 90% d'accuracy 
