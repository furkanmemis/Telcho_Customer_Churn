import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import torch
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

############
#EDA
############


def check_dataframe(df,head = 5):
    print("######################## HEAD ########################")
    print(df.head(head))
    print("######################## TAIL ########################")
    print(df.tail(head))
    print("######################## SHAPE ########################")
    print(df.shape)
    print("######################## COLUMN NAMES ########################")
    print(df.columns)
    print("######################## DESCRIBE ########################")
    print(df.describe().T)
    print("####################### INFO ########################")
    df.info()


check_dataframe(df)


def grab_columns(df, cat_n = 10, car_n = 20):
    """
    Notes:
    cat_col: categorical columns
    num_col: numerical columns
    car_col: cardina columns
    :param df:
    :param cat_n:
    :param car_n:
    :return: cat_col, num_col, car_col

    """



    cat_col = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if (df[col].dtypes != "O") & (df[col].nunique() < cat_n)]
    cat_but_car = [col for col in df.columns if (df[col].dtypes == "O") & (df[col].nunique() > car_n)]

    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in df.columns if (df[col].dtypes != "O") & (col not in num_but_cat)]

    return cat_col, num_col, cat_but_car

cat_col, num_col, car_col = grab_columns(df)

df[num_col].head()
df[cat_col].head()
df[car_col].head()

############
#Type Error
############
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")

check_dataframe(df)
cat_col, num_col, car_col = grab_columns(df)

df[num_col].head()
df[cat_col].head()
df[car_col].head()


df["Churn"] = df["Churn"].apply(lambda x: 1 if x== "Yes" else 0)
##################
#Numerical And Categorical Analysis
##################

def num_analysis(df,col,plot = False):
    qua = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,.95]

    print(df[col].describe(qua))
    print("###################################################################################")
    if plot:
        df[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show(block = True)


for col in num_col:
    num_analysis(df,col,plot = True)


def cat_analysis(df,col,plot = False):
    print(pd.DataFrame({col:df[col].value_counts(),"Ratio":100*df[col].value_counts()/len(df)}))
    print("###################################################################################")

    if plot:
       plt.figure(figsize= (10,7.5))
       sns.countplot(x=df[col],data= df)
       plt.xticks(rotation=30)
       plt.show(block = True)

for col in cat_col:
    cat_analysis(df,col,plot = True)


def target_col_and_cat_col_analysis(df,col,target):
    print(col)
    print(pd.DataFrame({target:df.groupby(col)[target].mean(),
                        "Count":df[col].value_counts(),
                        "Ratio":100*df[col].value_counts()/len(df)}))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

for col in cat_col:
    target_col_and_cat_col_analysis(df,col,"Churn")

####################
#Outlier Values
####################

def outlier_thresholds(df,col,qu1 = 0.1,qu3 = 0.9):
    q1 = df[col].quantile(qu1)
    q3 = df[col].quantile(qu3)

    ıqr = q3 - q1

    low = q1 - 1.5*ıqr
    up = q3 + 1.5*ıqr

    return low,up

def check_outlier(df,col):
    low,up = outlier_thresholds(df,col)

    if df[(df[col] < low) | (df[col] > up)].any(axis = None):
        return True
    else:
        return False

for col in num_col:
    print(col,"-",check_outlier(df,col))
#Out
"""
tenure - False
MonthlyCharges - False
TotalCharges - False
"""

###################
#Check Na Values
###################

df.isnull().sum().sort_values(ascending=False)



############################################
#Feature Extraction And Feature Interaction
############################################

df["AvgPay"] = df["TotalCharges"] / df["tenure"]

df["Years"] = df["tenure"].apply(lambda x: int(x/12))

df["AutoPay"] = df["PaymentMethod"].apply(lambda x: 1 if "automatic" in x else 0)

df.loc[(df["PaymentMethod"] == "Credit card (automatic)") | (df["PaymentMethod"] == "Bank transfer (automatic)"),"New_PaymentMethod"] = "Bank"
df.loc[(df["PaymentMethod"] == "Mailed check") | (df["PaymentMethod"] == "Electronic check"),"New_PaymentMethod"] = "Check"

df.loc[(df["tenure"] <= 24),"Cat_Years"] = "2Years"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 48),"Cat_Years"] = "4Years"
df.loc[(df["tenure"] > 48),"Cat_Years"] = "4+Years"

df["Annual_Contract"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

df.loc[(df["InternetService"] != "No") & (df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes") & (df["DeviceProtection"] == "Yes") & (df["TechSupport"] == "Yes") & (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes"),"MaxPackage"] = 1
df.loc[(df["InternetService"] == "No") | (df["OnlineSecurity"] == "No") | (df["OnlineBackup"] == "No") | (df["DeviceProtection"] == "No") | (df["TechSupport"] == "No") | (df["StreamingTV"] == "No") | (df["StreamingMovies"] == "No"),"MaxPackage"] = 0

df["Support"] = df.apply(lambda x: 0 if (x["OnlineBackup"] == "No") or (x["DeviceProtection"] == "No") or (x["TechSupport"] == "No") else 1, axis=1)


cat_col, num_col, car_col = grab_columns(df)

#Numerical Analysis

for col in num_col:
    num_analysis(df,col,plot = True)

#Categorical Analysis

for col in cat_col:
    cat_analysis(df,col,plot = True)

for col in cat_col:
    target_col_and_cat_col_analysis(df,col,"Churn")


#Outlier Check
for col in num_col:
    print(col,"-",check_outlier(df,col))


################################
#ENCODİNG
################################

#Label Encoding/Binary Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df =label_encoder(df,col)


#One Hot Encoding
onehot = [col for col in cat_col if col not in binary_cols and col not in ["Churn"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, onehot, drop_first=True)

#Check Na

df.isnull().sum().sort_values(ascending = False)
df = df.dropna()

################################################################################################################################
#MODEL
################################################################################################################################

y = df["Churn"]
X = df.drop(["Churn","customerID"],axis = 1)

#Scaling

#KNN
######################
knn_model = KNeighborsClassifier().fit(X,y)

y_pred_knn = knn_model.predict(X)

print(classification_report(y,y_pred_knn))
"""
              precision    recall  f1-score   support
           0       0.86      0.93      0.89      5163
           1       0.74      0.58      0.65      1869
    accuracy                           0.83      7032
   macro avg       0.80      0.75      0.77      7032
weighted avg       0.83      0.83      0.83      7032
"""
random_user = X.sample(random_state = 17)
#index = 184

df["Churn"].loc[184]

#Real: 1

knn_test_result = knn_model.predict(random_user)
#result: 1

#KNN Holdaout

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=17)
knn_model_hol = KNeighborsClassifier().fit(X_train,y_train)

y_pred_knn_hol = knn_model_hol.predict(X_test)

#Test
knn_test_result_hol = knn_model_hol.predict(random_user)
#result: 0


print(classification_report(y_test,y_pred_knn_hol))


"""
              precision    recall  f1-score   support
           0       0.83      0.89      0.86      1562
           1       0.60      0.47      0.53       548
    accuracy                           0.78      2110
   macro avg       0.72      0.68      0.69      2110
weighted avg       0.77      0.78      0.77      2110
"""

#Logistic Regression
######################

logistic_model = LogisticRegression(max_iter=10000).fit(X,y)

cv_results = cross_validate(logistic_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


for i in cv_results:
    print(i,": ",cv_results[i].mean())


"""
test_accuracy :  0.8054604037322026
test_precision :  0.6652162737960537
test_recall :  0.5398531920689309
test_f1 :  0.5959609738674623
test_roc_auc :  0.8448345207925906
"""

log_model_test = logistic_model.predict(random_user)
#real:1
#result:0

random_user1 = X.sample(random_state = 134)
df["Churn"].loc[897]
#real:0
log_model_test_1 = logistic_model.predict(random_user1)
#result:0


#CART
##########################

cart_model = DecisionTreeClassifier().fit(X,y)
y_pred_cart = cart_model.predict(X)

print(classification_report(y,y_pred_cart))

cart_model_test = cart_model.predict(random_user)
#real:1
#result:1

cart_model_test_1 = cart_model.predict(random_user1)
#real:0
#result:0

"""
              precision    recall  f1-score   support
           0       1.00      1.00      1.00      5163
           1       1.00      0.99      1.00      1869
    accuracy                           1.00      7032
   macro avg       1.00      1.00      1.00      7032
weighted avg       1.00      1.00      1.00      7032

"""

#CART Holdaout

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=13)

cart_model_hol = DecisionTreeClassifier().fit(X_train,y_train)

y_pred_cart_hol = cart_model_hol.predict(X_train)

print(classification_report(y_train,y_pred_cart_hol))


"""
              precision    recall  f1-score   support
           0       1.00      1.00      1.00      3592
           1       1.00      0.99      1.00      1330
    accuracy                           1.00      4922
   macro avg       1.00      1.00      1.00      4922
weighted avg       1.00      1.00      1.00      4922
"""

#CART CV

cv_result = cross_validate(cart_model,X,y,cv=5,scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

for i in cv_result:
    print(i,": ",cv_result[i].mean())

#GridSearchCv

cart_model.get_params()

cart_params = {"max_depth":range(1,11),
               "min_samples_split":range(2,12)}

cart_best_grid = GridSearchCV(cart_model,cart_params,cv=5,n_jobs=-1,verbose=1).fit(X,y)

cart_best_grid.best_params_


cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(cart_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

for i in cv_results:
    print(i,": ",cv_results[i].mean())


"""
fit_time :  0.03751349449157715
score_time :  0.009376335144042968
test_accuracy :  0.7891072982981859
test_f1 :  0.5389712210743981
test_roc_auc :  0.8312046841603002
"""
#Random Forest Classifier
##########################

rf_model = RandomForestClassifier(random_state=17)

rf_model.get_params()

cv_results = cross_validate(rf_model,X,y,cv=10,scoring=["accuracy","f1","roc_auc"])

for i in cv_results:
    print(i,": ",cv_results[i].mean())



rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}



rf_model_grid = GridSearchCV(rf_model,rf_params,cv=5,verbose=1,n_jobs=-1).fit(X,y)


rf_model_grid.best_params_

rf_final_model = rf_model.set_params(**rf_model_grid.best_params_,random_state = 17).fit(X,y)

cv_results = cross_validate(rf_final_model,X,y,cv=10,scoring=["accuracy","f1","roc_auc"])

for i in cv_results:
    print(i,": ",cv_results[i].mean())

"""
fit_time :  0.5755433559417724
score_time :  0.042852020263671874
test_accuracy :  0.8021860451959137
test_f1 :  0.5760841801836539
test_roc_auc :  0.8465125906110625
"""


#Gradient Boosting Decesion Tree
###############################

gbm_model = GradientBoostingClassifier(random_state=16).fit(X,y)

cv_results = cross_validate(gbm_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

for i in cv_results:
    print(i,": ",cv_results[i].mean())


"""
fit_time :  1.230377197265625
score_time :  0.02167959213256836
test_accuracy :  0.8021883065873638
test_f1 :  0.5842809152753712
test_roc_auc :  0.8446359794227668
"""



gbm_model.get_params()



gbm_params = {"learning_rate":[0.01,0.05,0.1],
              "max_depth":range(3,10),
              "min_samples_split":range(2,10),
              "n_estimators":[100,110,120,130],}


gbm_grid = GridSearchCV(gbm_model,gbm_params,cv=5,n_jobs=-1,verbose=1).fit(X,y)

gbm_grid.best_params_

gbm_final_model = gbm_model.set_params(**gbm_grid.best_params_,random_state = 17).fit(X,y)

cv_results = cross_validate(gbm_final_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

for i in cv_results:
    print(i,": ",cv_results[i].mean())
"""
fit_time :  1.1540412425994873
score_time :  0.01900057792663574
test_accuracy :  0.8044636601588684
test_f1 :  0.584695959193675
test_roc_auc :  0.8465682105758197
"""

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

for i in cv_results:
    print(i,": ",cv_results[i].mean())

"""
fit_time :  0.42475185394287107
score_time :  0.028482961654663085
test_accuracy :  0.7936561856436168
test_f1 :  0.5747129749829504
test_roc_auc :  0.8345937363077169
"""


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])


for i in cv_results:
    print(i,": ",cv_results[i].mean())


"""
fit_time :  0.3077881336212158
score_time :  0.025224924087524414
test_accuracy :  0.8033256800735199
test_f1 :  0.5788829665578811
test_roc_auc :  0.8437469441323401

"""

# Hiperparametre yeni değerlerle
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

for i in cv_results:
    print(i, ": ", cv_results[i].mean())

"""
fit_time :  0.3700835704803467
score_time :  0.025726795196533203
test_accuracy :  0.8033258822732506
test_f1 :  0.5761069226750655
test_roc_auc :  0.8444537455108087
"""

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

for i in cv_results:
    print(i, ": ", cv_results[i].mean())


"""
fit_time :  5.513331174850464
score_time :  0.015665817260742187
test_accuracy :  0.8017608563563001
test_f1 :  0.5819663104035928
test_roc_auc :  0.8424884040465578


"""

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

for i in cv_results:
    print(i, ": ", cv_results[i].mean())

"""
fit_time :  1.276416015625
score_time :  0.004830789566040039
test_accuracy :  0.8064548220086319
test_f1 :  0.577189365727303
test_roc_auc :  0.8479700781077598
"""
