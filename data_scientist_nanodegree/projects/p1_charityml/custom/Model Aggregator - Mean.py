
# coding: utf-8

# In[5]:


# !pip install oolearning --upgrade


# In[3]:


import copy
import os
import oolearning as oo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.getcwd()

def column_log(x):
    return np.log(x + 1)


csv_file = '/home/shane/repos/udacity/data_scientist_nanodegree/projects/p1_charityml/census.csv'
target_variable = 'income'

explore = oo.ExploreClassificationDataset.from_csv(csv_file_path=csv_file,
                                                   target_variable=target_variable)
negative_class = '<=50K'
positive_class = '>50K'

explore.dataset.head(20)


# In[8]:


n_positive = np.sum(explore.dataset[target_variable] == positive_class)
n_negative = np.sum(explore.dataset.income == negative_class)
scale_pos_weight_calc = n_negative / n_positive
scale_pos_weight_calc


# In[5]:


def create_net_capital(x):
    temp = x.copy()
    temp['net capital'] = temp['capital-gain'] - temp['capital-loss']
    return temp


# In[6]:


global_transformations = [
    # kaggle test file has white space around values
    oo.StatelessColumnTransformer(columns=explore.categoric_features,
                                  custom_function=lambda x: x.str.strip()),
    oo.ImputationTransformer(),
    oo.StatelessColumnTransformer(columns=['capital-gain', 'capital-loss'],
                               custom_function=column_log),
    oo.StatelessTransformer(custom_function=create_net_capital),
    oo.CenterScaleTransformer(),
    oo.DummyEncodeTransformer(oo.CategoricalEncoding.ONE_HOT)
]


# In[15]:


model_infos = [oo.ModelInfo(model=oo.RandomForestClassifier(extra_trees_implementation=True),
                            hyper_params=oo.RandomForestHP(
                                criterion='gini',
                                num_features=None,
                                max_features=1.0,
                                n_estimators=3750,
                                max_depth=14,
                            )),
               oo.ModelInfo(model=oo.RandomForestClassifier(),
                            hyper_params=oo.RandomForestHP(
                                criterion='gini',
                                num_features=None,
                                max_features=0.2,
                                n_estimators=1815,
                                max_depth=20,
                                min_samples_split=16,
                                min_samples_leaf=2,
                                min_weight_fraction_leaf=0.0,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0,
                            )),
               oo.ModelInfo(model=oo.AdaBoostClassifier,
                            hyper_params=oo.AdaBoostClassifierHP(
                                n_estimators=4250,
                                learning_rate=0.45,
                                algorithm='SAMME.R',
                                # Tree-specific hyper-params
                                criterion='gini',
                                splitter='best',
                                max_features=0.3,
                                max_depth=2,
                                min_samples_split=0.7,
                                min_samples_leaf=0.004,
                                min_weight_fraction_leaf=0.,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.,
                                class_weight=None,
                            )),
               oo.ModelInfo(model=oo.LogisticClassifier,
                            hyper_params=oo.LogisticClassifierHP(
                                penalty='l2',
                                regularization_inverse=0.245
                            )),
               oo.ModelInfo(model=oo.SvmLinearClassifier,
                            hyper_params=oo.SvmLinearClassifierHP(
                                penalty='l2',
                                penalty_c=10,
                            )),
               oo.ModelInfo(model=oo.XGBoostClassifier,
                            hyper_params=oo.XGBoostTreeHP(
                                objective=oo.XGBObjective.BINARY_LOGISTIC,
                                learning_rate=0.045,
                                n_estimators=3000,
                                max_depth=3,
                                min_child_weight=5,
                                gamma=0.15,
                                subsample=1,
                                colsample_bytree=0.4,
                             reg_alpha=0,
                             reg_lambda=2,
                             scale_pos_weight=scale_pos_weight_calc,
                         )),
]


# In[19]:


# use the ideal threshold for the evaluator in order to view ROC
evaluator = oo.TwoClassProbabilityEvaluator(converter=oo.TwoClassThresholdConverter(positive_class=positive_class))

trainer = oo.ModelTrainer(model=oo.ModelAggregator(base_models=model_infos,
                                                   aggregation_strategy=oo.SoftVotingAggregationStrategy()),
                          model_transformations=[t.clone() for t in global_transformations],
                          splitter=oo.ClassificationStratifiedDataSplitter(holdout_ratio=0.2),  # don't split, train on all data
                          evaluator=evaluator)
predictions = trainer.train_predict_eval(data=explore.dataset,
                                         target_variable=target_variable)

trainer.training_evaluator.all_quality_metrics


# In[ ]:


trainer.holdout_evaluator.all_quality_metrics

