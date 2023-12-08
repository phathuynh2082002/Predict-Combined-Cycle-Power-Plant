# ====================================================== KHAI BAO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from sklearn import  linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree

# ====================================================== DOC VA CHUAN HOA TAP DU LIEU

dulieuzin= pd.read_excel('dataset.xlsx')
mms= pp.MinMaxScaler()
dulieu=mms.fit_transform(dulieuzin)


X=dulieu[:,0:4]
y=dulieu[:,4:5]
y = y.reshape(-1,)

# print("DU LIEU SAU KHI CHUAN HOA : ")
# print(X)
# print(y)
#
# print("\n\n KNN 10 LAN LAP======================================================\n")
# def KNN(X,y,i):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3.0, random_state=i)
#     Mohinh_KNN= KNeighborsRegressor(n_neighbors=7)
#     Mohinh_KNN.fit(X_train,y_train)
#     y_pred= Mohinh_KNN.predict(X_test)
#     err = mean_squared_error(y_test,y_pred)
#     rmse = round(np.sqrt(err),3)
#     print("RMSE OF KNN : ",rmse)
#     # print('ACCURACY OF KNN : ',round( (np.mean(np.abs((y_test - y_pred) / y_test)) * 100),3),'%')
#     return rmse
#
# tb1=0
# for i in range(1,11):
#     tb1 += KNN(X,y,i)
# tb1 /=10
#
# print("\n\n DECISION TREE BY BAGGING 10 LAN LAP======================================================\n")
#
# def DTB(X,y,i):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3.0, random_state=i)
#     tree = DecisionTreeRegressor(random_state=0)
#     bagging_regtree=BaggingRegressor(base_estimator=tree,n_estimators=10,random_state=42)
#     bagging_regtree.fit(X_train,y_train)
#     y_pred= bagging_regtree.predict(X_test)
#     err=mean_squared_error(y_test,y_pred)
#     rmse=round(np.sqrt(err),3)
#     print("RMSE OF DECISION TREE BAGGING : ",rmse)
#     return rmse
#
# tb3=0
# for i in range(1,11):
#     tb3 += DTB(X,y,i)
# tb3 /= 10
#
# print("\n\nLINEAR REGRESSION BY BAGGING 10 LAN LAP======================================================\n")
# def LRB(X,y,i):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3.0, random_state=i)
#     lm=linear_model.LinearRegression()
#     bagging_reg=BaggingRegressor(base_estimator=lm,n_estimators=10,random_state=42)
#     bagging_reg.fit(X_train,y_train)
#     y_pred=bagging_reg.predict(X_test)
#     err=mean_squared_error(y_test,y_pred)
#     rmse=round(np.sqrt(err),3)
#     print("RMSE OF LINEAR REGRESSION BAGGING : ",rmse)
#     return rmse
#
# tb4=0
# for i in range(1,11):
#     tb4 += LRB(X,y,i)
# tb4 /= 10
#
# print("\n\nRMSE TRUNG BINH 10 LAN LAP CUA TUNG MO HINH ======================================================\n")
# print("KNN : ",tb1)
# print("DECISION TREE BY BAGGING : ",tb3)
# print("LINEAR REGRESSION BY BAGGING : ",tb4)
#

print("\n\nCAY HOI QUY======================================================\n")
def Graph(X,y):
    regr = DecisionTreeRegressor(max_depth=3, random_state=1234)
    model = regr.fit(X, y)
    text_representation = tree.export_text(regr)
    print(text_representation)
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(regr, feature_names='PE', filled=True)
    # dot_data = tree.export_graphviz(regr, out_file=None,
    #                                 feature_names=y,
    #                                 filled=True)
    # graphviz.Source(dot_data, format="png")
    plt.show()
Graph(X,y)

#DO THI BIEU DIEN TUONG QUAN GIUA CAC THUOC TINH VOI SAN LUONG DIEN

# lm = linear_model.LinearRegression()

# ====================================================== NHIET DO

# AT=dulieu[:,0:1]
# lm.fit(AT,y)
#
# plt.axis([0,1,0,1])
# plt.plot(AT,y,"ro",color="blue")
# predict=[lm.coef_*i+lm.intercept_ for i in AT]
# plt.plot(AT,predict,color="green")
# plt.xlabel("Gia tri thuoc tinh AT")
# plt.ylabel("Gia tri thuoc tinh PE")
# plt.show()

# ====================================================== CHAN KHONG KHI THAI

# V=dulieu[:,1:2]
# lm.fit(V,y)
#
# plt.axis([0,1,0,1])
# plt.plot(V,y,"ro",color="blue")
# predict=[lm.coef_*i+lm.intercept_ for i in V]
# plt.plot(V,predict,color="green")
# plt.xlabel("Gia tri thuoc tinh V")
# plt.ylabel("Gia tri thuoc tinh PE")
# plt.show()

# ====================================================== AP SUAT MOI TRUONG XUNG QUANH

# AP=dulieu[:,2:3]
# lm.fit(AP,y)
#
# plt.axis([0,1,0,1])
# plt.plot(AP,y,"ro",color="blue")
# predict=[lm.coef_*i+lm.intercept_ for i in AP]
# plt.plot(AP,predict,color="green")
# plt.xlabel("Gia tri thuoc tinh AP")
# plt.ylabel("Gia tri thuoc tinh PE")
# plt.show()

# ====================================================== DO AM TUONG DOI
# RH=dulieu[:,3:4]
# lm.fit(RH,y)
#
# plt.axis([0,1,0,1])
# plt.plot(RH,y,"ro",color="blue")
# predict=[lm.coef_*i+lm.intercept_ for i in RH]
# plt.plot(RH,predict,color="green")
# plt.xlabel("Gia tri thuoc tinh RH")
# plt.ylabel("Gia tri thuoc tinh PE")
# plt.show()

# ======================================================
