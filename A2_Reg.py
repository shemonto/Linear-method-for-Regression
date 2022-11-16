#Group 15
#Author: Shemonto Das #202193149 & Farhan Anjum Haque #201755113
#Date: 27/02/2022
from cgi import test
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import xgboost as xgb
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from numpy import mean


df_train = pd.read_csv(sys.argv[1], sep="\t")
test = pd.read_csv(sys.argv[2], sep="\t")
#print(df_train)
#print(test)

a1 = df_train.iloc[:,:-1]
b1 = df_train.iloc[:,-1]

###############################################################
###################  RSS  #########################
###############################################################

def rss_score(y, y_pred):
    return np.sum((y - y_pred)**2)

###############################################################
###################  BASE-LINE MODEL  #########################
###############################################################
reg = LinearRegression().fit(a1, b1)
model = LinearRegression()
scores = cross_val_score(reg, a1, b1, scoring='r2', cv=10, n_jobs=-1)

#print('R^2 of base-line model: %.2f ' % mean(scores))
#print("Coefficient of base-line model: ", reg.coef_)
rp = reg.predict(a1)
val1 = rss_score(b1,rp)
#print('RSS of base-line model: %.2f ' % val1)

#reg.predict(test)

###############################################################
################### RIDGE-REGRESSION  #########################
###############################################################
# find optimal alpha with grid search
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)
ridge = Ridge()
grid_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid, cv = 10, scoring='r2', verbose=1, n_jobs=-1)
grid_result_ridge = grid_ridge.fit(a1, b1)
#print('Best Score of Ridge-regression: ', grid_result_ridge.best_score_)
#print('Best Params of Ridge-regression: ', grid_result_ridge.best_params_)
#print('RSS: %.3f ' % rss_score(b1,ridge_predict))

###############################################################
################### LASSO-REGRESSION  #########################
###############################################################
# find optimal alpha with grid search
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)
lasso = Lasso()
grid_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='r2', cv=10, verbose=1, n_jobs=-1)
grid_result_lasso = grid_lasso.fit(a1,b1)
lasso_predict = grid_result_lasso.predict(a1)
#print('Best Score of Lasso-regression: %.2f' %  grid_result_lasso.best_score_)
#print('Best Params of Lasso-regression: ', grid_result_lasso.best_params_)
#print('RSS of Lasso-regression: %.3f ' % rss_score(b1,lasso_predict))

###############################################################
################### ELASTIC-NET-REGRESSION  #########################
###############################################################
# find optimal alpha with grid search
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
param_grid = dict(alpha=alpha, l1_ratio=l1_ratio)
elastic_net = ElasticNet()
grid_elastic = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv = 10, scoring='r2', verbose=1, n_jobs=-1)
grid_result_elastic = grid_elastic.fit(a1,b1)
elastic_predict = grid_result_elastic.predict(a1)
#print('Best Score of elastic-net regression: %.2f' % grid_result_elastic.best_score_)
#print('Best Params of elastic-net regression: ', grid_result_elastic.best_params_)
#print('RSS of elastic-net regression: %.3f ' % rss_score(b1,elastic_predict))

###############################################################
################### KNN-REGRESSION  #########################
###############################################################
# find optimal alpha with grid search CV
neighbor = [3,5,7,9,11]
l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
 
param_grid = dict(n_neighbors=neighbor)
neigh = KNeighborsRegressor()
grid_knn = GridSearchCV(neigh, param_grid,cv =10, scoring='r2', verbose=1, n_jobs=-1)
grid_result_knn = grid_knn.fit(a1,b1)
knn_predict = grid_result_knn.predict(a1)
#print('Best Score of knn-regression: %.2f' % grid_result_knn.best_score_)
#print('Best Params of knn-regression: ', grid_result_knn.best_params_)
#print('RSS of knn-regression: %.3f ' % rss_score(b1,knn_predict))

###############################################################
################### XGB-BOOST  #########################
###############################################################
params = {
        
        'gamma': [0.5, 1, 1.5, 2, 5],
        'alpha': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

# find optimal alpha with grid search CV
#learning_rate = [3,5,7,9,11]
#l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
 
param_grid = params
neigh =XGBRegressor()
grid_xgb = GridSearchCV(neigh, param_grid,cv =10, scoring='r2', verbose=1, n_jobs=-1)
grid_result_xgb = grid_xgb.fit(a1,b1)
xgb_predict = grid_result_xgb.predict(a1)
#print('Best Score of XGB-boost: %.2f' % grid_result_xgb.best_score_)
#print('Best Params of XGB-boost: ', grid_result_xgb.best_params_)
#print('RSS of XGB-boost: %.3f ' % rss_score(b1,xgb_predict))

###############################################################
################### BEST-MODEL  #########################
###############################################################
g_reg = xgb.XGBRegressor( colsample_bytree = 0.6, max_depth = 5, alpha = 0.6, gama = 2,scoring='r2').fit(a1,b1)
xg = g_reg.predict(a1)
#print('R^2 of XGB-boost:',g_reg.score(a1,b1))
#print('RSS of XGB-boost: %.3f ' % rss_score(b1,xg))


##########################################################################
########################### WRITE FILE ###################################
das = g_reg.predict( test)
das2 = (np.around(das, 2))
a_file = open("A2_predictions_group15.txt", "w")
np.savetxt("A2_predictions_group15.txt", np.array(das2))

############################################################################
print('CV R^2 of base-line model: %.2f ' % mean(scores))
print("Coefficient of base-line model: ", reg.coef_)
print('RSS of base-line model: %.2f ' % val1)

print('Best Score of Ridge-regression: ', grid_result_ridge.best_score_)
print('Best Params of Ridge-regression: ', grid_result_ridge.best_params_)

print('Best Score of Lasso-regression: %.2f' %  grid_result_lasso.best_score_)
print('Best Params of Lasso-regression: ', grid_result_lasso.best_params_)

print('Best Score of elastic-net regression: %.2f' % grid_result_elastic.best_score_)
print('Best Params of elastic-net regression: ', grid_result_elastic.best_params_)

print('Best Score of knn-regression: %.2f' % grid_result_knn.best_score_)
print('Best Params of knn-regression: ', grid_result_knn.best_params_)

print('Best CV Score of XGB-boost: %.2f' % grid_result_xgb.best_score_)
print('Best Params of XGB-boost: ', grid_result_xgb.best_params_)

print('Our best model is XGB-regressor.')
print('R^2 of XGB-boost:',g_reg.score(a1,b1))
print('RSS of XGB-boost: %.3f ' % rss_score(b1,xg))