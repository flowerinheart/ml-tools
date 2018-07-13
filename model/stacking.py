from sklearn.model_selection import KFold
import xgboost as xgb
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn import metrics

"""
stacking 的思想很好理解，但是在实现时需要注意不能有泄漏（leak）的情况，也就是说对于训练样本中的每一条数据，
基模型输出其结果时并不能用这条数据来训练。否则就是用这条数据来训练，同时用这条数据来测试，这样会造成最终预测时的过拟合现象，即经过stacking后在训练集上进行验证时效果很好，但是在测试集上效果很差。

为了解决这个泄漏的问题，需要通过 K-Fold 方法分别输出各部分样本的结果，这里以 5-Fold 为例，具体步骤如下

(1) 将数据划分为 5 部分，每次用其中 1 部分做验证集，其余 4 部分做训练集，则共可训练出 5 个模型
(2) 对于训练集，每次训练出一个模型时，通过该模型对没有用来训练的验证集进行预测，将预测结果作为验证集对应的样本的第二层输入，
则依次遍历5次后，每个训练样本都可得到其输出结果作为第二层模型的输入
(3) 对于测试集，每次训练出一个模型时，都用这个模型对其进行预测，则最终测试集的每个样本都会有5个输出结果，对这些结果取平均作为该样本的第二层输入
"""

class BasicModel(object):
    """Parent class of basic models."""

    def train(self, x_train, y_train, x_val, y_val):
        """Return a trained model and eval metric of validation data."""
        pass

    def predict(self, model, x_test):
        """Return the predicted result of test data."""
        pass

    def get_oof(self, x_train, y_train, x_test, n_folds=5):
        """K-fold stacking."""
        num_train, num_test = x_train.shape[0], x_test.shape[0]
        oof_train = np.zeros((num_train,))
        oof_test = np.zeros((num_test,))
        oof_test_all_fold = np.zeros((num_test, n_folds))
        aucs = []
        KF = KFold(n_splits=n_folds, random_state=2017)
        for i, (train_index, val_index) in enumerate(KF.split(x_train)):
            print('{0} fold, train {1}, val {2}'.format(i,
                                                        len(train_index),
                                                        len(val_index)))
            x_tra, y_tra = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]
            model, auc = self.train(x_tra, y_tra, x_val, y_val)
            aucs.append(auc)
            oof_train[val_index] = self.predict(model, x_val)
            oof_test_all_fold[:, i] = self.predict(model, x_test)
        oof_test = np.mean(oof_test_all_fold, axis=1)
        print('all aucs {0}, average {1}'.format(aucs, np.mean(aucs)))
        return oof_train, oof_test


class XGBClassifier(BasicModel):
    """Xgboost model for stacking."""

    def __init__(self):
        """Set parameters."""
        self.num_rounds = 1000
        self.early_stopping_rounds = 15
        self.params = {
            'objective': 'binary:logistic',
            'eta': 0.1,
            'max_depth': 8,
            'eval_metric': 'auc',
            'seed': 0,
            'silent': 0
         }

    def train(self, x_train, y_train, x_val, y_val):
        """T."""
        print('train with xgb model')
        xgbtrain = xgb.DMatrix(x_train, y_train)
        xgbval = xgb.DMatrix(x_val, y_val)
        watchlist = [(xgbtrain, 'train'), (xgbval, 'val')]
        model = xgb.train(self.params,
                          xgbtrain,
                          self.num_rounds,
                          watchlist,
                          early_stopping_rounds=self.early_stopping_rounds)
        return model, float(model.eval(xgbval).split()[1].split(':')[1])

    def predict(self, model, x_test):
        print('test with xgb model')
        xgbtest = xgb.DMatrix(x_test)
        return model.predict(xgbtest)


class LGBClassifier(BasicModel):
    def __init__(self):
        self.num_boost_round = 2000
        self.early_stopping_rounds = 15
        self.params = {
            'task': 'train',
            'boosting_type': 'dart',
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'num_leaves': 80,
            'learning_rate': 0.05,
            # 'scale_pos_weight': 1.5,
            'feature_fraction': 0.5,
            'bagging_fraction': 1,
            'bagging_freq': 5,
            'max_bin': 300,
            'is_unbalance': True,
            'lambda_l2': 5.0,
            'verbose': -1
            }

    def train(self, x_train, y_train, x_val, y_val):
        print('train with lgb model')
        lgbtrain = lgb.Dataset(x_train, y_train)
        lgbval = lgb.Dataset(x_val, y_val)
        model = lgb.train(self.params,
                          lgbtrain,
                          valid_sets=lgbval,
                          verbose_eval=self.num_boost_round,
                          num_boost_round=self.num_boost_round,
                          early_stopping_rounds=self.early_stopping_rounds)
        return model, model.best_score['valid_0']['auc']

    def predict(self, model, x_test):
        print('test with lgb model')
        return model.predict(x_test, num_iteration=model.best_iteration)


def lgb_xgboost_lr_stacking(x_train, y_train,
                            x_test, y_test):
    """Return a basic stacking model."""
    lgb_classifier = LGBClassifier()
    lgb_oof_train, lgb_oof_test = lgb_classifier.get_oof(x_train,
                                                         y_train, x_test)
    print(lgb_oof_train.shape, lgb_oof_test.shape)

    xgb_classifier = XGBClassifier()
    xgb_oof_train, xgb_oof_test = xgb_classifier.get_oof(x_train,
                                                         y_train, x_test)
    print(xgb_oof_train.shape, xgb_oof_test.shape)

    input_train = [xgb_oof_train, lgb_oof_train]
    input_test = [xgb_oof_test, lgb_oof_test]

    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    print(stacked_train.shape, stacked_test.shape)


    # use LR as the model of the second layer

    # split for validation
    n = int(stacked_train.shape[0] * 0.8)
    x_tra, y_tra = stacked_train[:n], y_train[:n]
    x_val, y_val = stacked_train[n:], y_train[n:]
    model = LinearRegression()
    model.fit(x_tra,y_tra)
    y_pred = model.predict(x_val)
    print(metrics.roc_auc_score(y_val, y_pred))

    # predict on test data
    final_model = LinearRegression()
    final_model.fit(stacked_train, y_train)
    test_prediction = final_model.predict(stacked_test)
