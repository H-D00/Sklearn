from sklearn.ensemble import RandomForestClassifier

from sklearn import datasets

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

wine=load_wine()

X=wine.data

y=wine.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

rfc=RandomForestClassifier(n_estimators=10,n_jobs=-1,random_state=50,min_samples_leaf=20)   

rfc.fit(X_train,y_train)

print(rfc.score(X_test,y_test))

#feature importance

imp=rfc.feature_importances_   

print(imp)

names=wine.feature_names       

print(names)


imp,names=zip(*sorted(zip(imp,names)))

plt.barh(range(len(names)),imp,align='center') 

plt.yticks(range(len(names)),names)

plt.xlabel('importances features')

plt.ylabel('features')

plt.title('wine importances feature')

plt.show




![image](https://github.com/H-D00/Sklearn/blob/fe5fcaeabfaf5b3ff5ae1f7dac541580103730a7/RandomForest/%5Bouput%5D%20RandomForest.jpg)
