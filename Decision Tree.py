# Decision Tree====================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
dulieu= pd.read_excel('dataset.xlsx')
X=dulieu.iloc[:,0:4]
y=dulieu.PE
# plt.xlabel("RH")
# plt.ylabel("PE")
# plt.scatter(dulieu.RH,dulieu.PE)
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=1/3.0,random_state=5)
print("train size:",len(y_train))
print("test size",len(y_test))

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(random_state=0)
tree.fit(X_train,y_train)
y_pred= tree.predict(X_test)
err=mean_squared_error(y_test,y_pred)
print(np.sqrt(err))

bagging_regtree=BaggingRegressor(base_estimator=tree,n_estimators=10,random_state=42)
bagging_regtree.fit(X_train,y_train)
y_pred= bagging_regtree.predict(X_test)

err=mean_squared_error(y_test,y_pred)
print(np.sqrt(err))




# Decision Tree====================================