# -*- coding: utf-8 -*-
"""
Created on Tue May 10 03:06:43 2022

@author: Aryan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  7 01:14:48 2022

@author: Aryan
"""



import random       
         
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv('hindi char.csv')
df['label'].value_counts()
df['label'].value_counts().plot.bar()


'''
sample = df.drop('label', axis=1).values[128].copy()
print(sample)
        
plt.figure(figsize=(10, 10))
plt.title(f"image is {df['label'].values[0]}")
plt.imshow(sample.reshape(32, 32), cmap="binary")
plt.show()
'''





from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = df.drop('label', axis=1).values
y = df['label'].values

print(X.shape)
print(y.shape)









from sklearn.preprocessing import MinMaxScaler
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import ExtraTreesClassifier

mms = MinMaxScaler()

mms.fit(X)

X_norm = mms.transform(X)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
#model=AdaBoostClassifier()
#model=ExtraTreesClassifier()
model.fit(X_norm, y)

y_train_pred = model.predict(X_norm)

print(f"Accuracy on training data: {accuracy_score(y_train_pred, y)}")



from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
confusion_matrix(y,y_train_pred)





















run = False

def draw(event, x, y, flag, param):
	global run

	if event == cv2.EVENT_LBUTTONDOWN:
		run = True
		cv2.circle(win, (x,y), 10 , (255,255,255), 10)

	if event == cv2.EVENT_LBUTTONUP:
		run = False

	if event == cv2.EVENT_MOUSEMOVE:
		if run == True:
			cv2.circle(win, (x,y), 10 , (random.randint(100,180),random.randint(100,180),random.randint(100,180)), 10)

	
cv2.namedWindow('window')
cv2.setMouseCallback('window', draw)

win = np.zeros((500,500,3), dtype='float64')


while True:

	cv2.imshow('window', win)

	k = cv2.waitKey(1)

	if k == ord('c'):
		win = np.zeros((500,500,3), dtype='float64')

	if k == ord('q'):
        
		cv2.destroyAllWindows()
		break
cv2.imwrite('s1.jpg',win)


image=cv2.imread('s1.jpg')

grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

image2=cv2.resize(grey_image,None,fx=32/image.shape[0],
                  fy=32/image.shape[1],interpolation = cv2.INTER_CUBIC)

tval1,image5=cv2.threshold(image2,100,255,cv2.THRESH_BINARY)

'''
cv2.imshow('original',image)

cv2.imshow('grey',grey_image)
cv2.imshow('compressed',image2)
cv2.imshow('comp edit',image5)

cv2.waitKey()
cv2.destroyAllWindows()
'''


x= []
for i in range(0,1024):
    x.append('pixel'+str(i))   
    
x = np.array(x)

#k = image5.reshape([1, 784])
k= image5.flatten()
k = k.astype('int64')

import random

for i in range(0,len(k)):
    if k[i]==255:
        if k[i+1]!=255:
            k[i+1]=random.randint(100,180)
    if k[i]==255 and k[i-1]!=255:
        k[i-1]=random.randint(100,180)
       

plt.imshow(k.reshape(32, 32), cmap="binary")
plt.show()


#x.tofile('data2.csv', sep = ',')

#np.savetxt("data2.csv", x, delimiter=",", fmt="%s")
#pd.DataFrame(x).to_csv('sample.csv')
x.tofile('data2.csv', sep = ',') 

df1 = pd.read_csv('data2.csv')
df1.loc[2] = k

df1.to_csv('data2.csv')

df1 = pd.read_csv('data2.csv')
first_column = df1.columns[0]
# Delete first
df1 = df1.drop([first_column], axis=1)
df1.to_csv('data2.csv', index=False)


df_test = pd.read_csv('data2.csv')
df_test.head()
X_test = df_test.values
X_norm_test = mms.transform(X_test)
y_test_pred = model.predict(X_norm_test)

y_test_pred.shape

'''
l=['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']

print('The image is ',y_test_pred[0],' or ',l[y_test_pred[0]])
'''


p=['character_01_ka',
   'character_02_kha',
   'character_03_ga',              
   'character_04_gha',             
   'character_05_kna',           
   'character_06_cha',             
   'character_07_chha',          
   'character_08_ja',            
   'character_09_jha',            
   'character_10_yna',          
   'character_11_taamatar',       
   'character_12_thaa',           
   'character_13_daa',            
   'character_14_dhaa',           
   'character_15_adna',            
   'character_16_tabala',         
   'character_17_tha',             
   'character_18_da',              
   'character_19_dha',             
   'character_20_na',              
   'character_21_pa',
   'character_22_pha', 
   'character_23_ba', 
   'character_24_bha', 
   'character_25_ma',
   'character_26_yaw',
   'character_27_ra',            
   'character_28_la',              
   'character_29_waw',             
   'character_30_motosaw',        
   'character_31_petchiryakha',    
   'character_32_patalosaw',       
   'character_33_ha'            
 
      ]


mm=['क',	'ख'	,'ग',	'घ','ङ','च', 'छ',	'ज','झ',	'ञ',	'ट',	'ठ',	'ड',	'ढ',	'ण',	'त',	'थ',	'द',	'ध',	'न',	'प',	'फ',	'ब',	'भ',	'म',	'य',	'र',	'ल',	'व',	'श',	'ष',	'स',	'ह'  ]


index=p.index(y_test_pred[0])

print('यह वर्णमाला है  :  '   ,mm[index])







from sklearn.metrics import confusion_matrix

x=np.array([[37,0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	3,	0,	0,	0,	0],
            [2,	34,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	0,	0,	0,	0,	0,	0,	2,	5,	4,	0,	0,	0,	2,	3,	0,	0],
            [0, 0,	32,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	0,	0,	0,	0],
            [0,	0,	0,	29,	0,	0,	6,	2,	0,	0,	0,	0,	0,	0,	5,	0,	0,	0,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	5,	0,	0],
            [0,	0,	0,	0,	35,	0,	0,	0,	0,	0,	0,	0,	12,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [1,	2,	0,	1,	0,	36,	0,	0,	0,	3,	0,	0,	0,	0,	4,	6,	3,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [0,	1,	0,	11,	0,	2,	39,	0,	4,	0,	0,	3,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	5,	0,	0,	0],
            [0,	0,	0,	0,	0,	3,	0,	33,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	6,	0,	0,	0,	0,	0,	0,	0,	6,	0,	0,	0,	0,	0],
            [0,	0,	0,	2,	0,	0,	3,	0,	24,	0,	0,	0,	0,	0,	0,	0,	0,	2,	0,	0,	0,	0,	0,	5,	0,	3,	0,	0,	0,	3,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	1,	40,	0,	0,	0,	0,	0,	0,	0,	0,	0,	3,	0,	0,	0,	0,	0,	0,	0,	4,	0,	0,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	41,	18,	3,	0,	0,	0,	0,	8,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	7],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	3,	29,	0,	15,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	3],
            [0,	0,	0,	0,	12,	0,	0,	0,	0,	0,	0,	0,	32,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	4],
            [0,	0,	0,	0,	3,	0,	0,	0,	0,	0,	0,	0,	0,	28,	0,	0,	0,	3,	0,	0,	0,	0,	4,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	38,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [0,	0,	5,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	35,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [0,	0,	4,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	40,	0,	9,	0,	0,	0,	0,	0,	0,	7,	4,	0,	0,	3,	5,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	4,	0,	1,	7,	0,	0,	0,	37,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	3],
            [0,	0,	0,	5,	0,	0,	2,	0,	4,	0,	0,	0,	0,	0,	0,	0,	0,	0,	39,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [0,	2,	0,	0,	0,	0,	0,	4,	0,	5,	0,	0,	0,	0,	0,	5,	0,	0,	0,	41,	0,	0,	0,	0,	0,	0,	0,	4,	0,	0,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	38,	3,	0,	0,	0,	3,	7,	0,	7,	0,	6,	0,	0],
            [5,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	44,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	34,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [3,	0,	4,	0,	0,	0,	0,	0,	10,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	0,	38,	7,	2,	0,	0,	0,	0,	0,	8,	0],
            [2,	0,	0,	0,	0,	0,	0,	0,	7,	0,	0,	0,	0,	0,	2,	0,	0,	0,	0,	0,	0,	1,	0,	5,	32,	0,	0,	3,	0,	0,	0,	1,	0],
            [0,	4,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	4,	0,	0,	0,	3,	0,	0,	0,	0,	31,	0,	0,	0,	0,	0,	7,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	39,	0,	3,	0,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	11,	0,	2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	33,	0,	0,	0,	0,	0],
            [0,	0,	0,	0,	0,	9,	0,	0,	0,	0,	0,	0,	0,	0,	0,	2,	0,	0,	0,	0,	5,	0,	12,	0,	0,	0,	0,	0,	35,	0,	0,	0,	0],
            [0,	0,	5,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	6,	0,	0,	0,	0,	32,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	3,	0,	0,	0,	0,	0,	0,	0,	0,	0,	31,	0,	0],
            [0,	7,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	5,	0,	34,	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	33]
            ])

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.heatmap(x, annot=True, cmap='Blues', xticklabels=p, yticklabels=p)

ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Numbers')
ax.set_ylabel('Actual Numbers ');

## Ticket labels - List must be in alphabetical order


## Display the visualization of the Confusion Matrix.
plt.show()







'''
dat123 = pd.read_csv('f.csv')
s=[]
print('Was I correct: 1 for Yes 0 for No :', end = ' ')
flag=int(input())
if flag==0:
    print('What was it then :', end = ' ')
    temp=input()
ind=mm.index(temp)
    

s.append(p[ind])
s.extend(k)

import csv   

with open('f.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(s)

'''

