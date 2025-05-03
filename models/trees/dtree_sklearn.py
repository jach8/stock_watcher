#########################################################################################################
# Random Forest Model, with AdaBoostClassifier. Baseline is KNN model. 
#########################################################################################################


import pandas as pd 
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

class Dtree():
    def __init__(self, train_x, train_y, test_x, test_y, target):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.target = target
    
    def sensitivity_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=1)

    def specificity_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=-1)

    def tree_size(self):
        train_x = self.train_x
        train_y = self.train_y
        tsize = np.arange(10, 100, 1)
        # Plot Training error for number of trees on the training data 
        ter = []
        for j in tqdm(range(5)):
            train_error = []
            for i in tqdm(tsize):
                rf = RandomForestClassifier(n_estimators=i)
                rf.fit(train_x, train_y)
                ada = AdaBoostClassifier(base_estimator=rf, n_estimators=i)
                ada.fit(train_x, train_y)
                train_error.append(1 - rf.score(train_x, train_y))
            ter.append(train_error)

        ter = np.array(ter)
        train_error = ter.mean(axis=0)
        self.tree_size_results = pd.DataFrame({'train_error': train_error})
        self.n_trees = tsize[np.argmin(train_error)]

    def var_select(self):
        train_x, train_y, test_x, test_y = self.train_x, self.train_y, self.test_x, self.test_y
        var_selection = range(1, train_x.shape[1] + 1)
        # Random forest
        ob = []
        te = []
        for j in tqdm(range(5)):
            OOB_error = []
            test_error = []
            for v in tqdm(var_selection):
                rf_v = RandomForestClassifier(max_features=v, oob_score=True)
                rf_v.fit(train_x, train_y)
                OOB_error.append(1 - rf_v.oob_score_)
                test_error.append(1 - rf_v.score(test_x, test_y))
            ob.append(OOB_error)
            te.append(test_error)

        OOB_error = np.array(ob).mean(axis=0)
        test_error = np.array(te).mean(axis=0)
        self.var_selection_results = pd.DataFrame({'OOB_error': OOB_error, 'test_error': test_error})
        self.nvar = var_selection[np.argmin(test_error)]
    
    def ada_boost_lr(self):
        train_x, train_y, test_x, test_y = self.train_x, self.train_y, self.test_x, self.test_y
        # Boosting
        param_grid = {
            'learning_rate' : np.arange(0.1, 1, 0.1)
        }
        rf = RandomForestClassifier(max_features = 8, n_estimators= 37)
        ada = AdaBoostClassifier(base_estimator=rf)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        print("Fitting Grid Search")
        grid_result = grid_search.fit(train_x, train_y)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']

        ada_train_error = []
        for mean, param in zip(means, params):
            ada_train_error.append(1 - mean)

        self.ada_train_results = pd.DataFrame({'train_error': ada_train_error})
        self.best_lr = grid_result.best_params_['learning_rate']
        self.best_ada = grid_result.best_estimator_

    def fit_mod(self):
        train_x, train_y, test_x, test_y = self.train_x, self.train_y, self.test_x, self.test_y
        self.ada_boost_lr()
        self.tree_size()
        self.var_select()
        rf = RandomForestClassifier(max_features = self.nvar, n_estimators= self.n_trees)
        ada = AdaBoostClassifier(base_estimator=rf, learning_rate=self.best_lr)

        bap = {}
        rfp = {}
        knp = {}
        for i in tqdm(range(50)):
            # Fit best model on test data 
            best_ada = self.best_ada
            best_ada.fit(train_x, train_y)
            best_ada_pred = best_ada.predict(test_x)
            bap['accuracy'] = accuracy_score(best_ada_pred, test_y)
            bap['sensitivity'] = self.sensitivity_score(best_ada_pred, test_y)
            bap['specificity'] = self.specificity_score(best_ada_pred, test_y)
            bap['precision'] = precision_score(best_ada_pred, test_y)
            
            # random forest
            rf = RandomForestClassifier(max_features = self.nvar, n_estimators= self.n_trees)
            rf.fit(train_x, train_y)
            rf_pred = rf.predict(test_x)
            rfp['accuracy'] = accuracy_score(rf_pred, test_y)
            rfp['sensitivity'] = self.sensitivity_score(rf_pred, test_y)
            rfp['specificity'] = self.specificity_score(rf_pred, test_y)
            rfp['precision'] = precision_score(rf_pred, test_y)

            # KNN 
            knn = KNeighborsClassifier(n_neighbors=np.sqrt(train_x.shape[0]).astype(int))
            knn.fit(train_x, train_y)
            knn_pred = knn.predict(test_x)
            #knp.append(accuracy_score(knn_pred, test_y))
            knp['accuracy'] = accuracy_score(knn_pred, test_y)
            knp['sensitivity'] = self.sensitivity_score(knn_pred, test_y)
            knp['specificity'] = self.specificity_score(knn_pred, test_y)
            knp['precision'] = precision_score(knn_pred, test_y)

        return pd.DataFrame({"AdaBoost": bap, "Random Forest": rfp, "KNN": knp})