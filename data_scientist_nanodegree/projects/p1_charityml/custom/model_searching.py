import oolearning as oo
import pandas as pd
import numpy as np
pd.set_option('display.width', 500)

import os
print(os.getcwd())

directory = 'projects/p1_charityml/'
#directory = '../'
csv_file = directory+'census.csv'
target_variable = 'income'
explore = oo.ExploreClassificationDataset.from_csv(csv_file_path=csv_file,
                                                   target_variable=target_variable)
negative_class = '<=50K'
positive_class = '>50K'

explore.dataset.head(20)


def column_log(x):
    return np.log(x + 1)


# TODO: reincorporate this back into the final models built; had to remove because the local function was
# making parallelization fail
transformer = oo.StatelessColumnTransformer(columns=['capital-gain', 'capital-loss'],
                                            custom_function=column_log)

transformed_dataset = transformer.fit_transform(data_x=explore.dataset)
# explore.dataset['capital-gain'].hist(bins=15)
# transformed_dataset['capital-gain'].hist(bins=15)

global_transformations = [oo.ImputationTransformer(),
                          # TODO: reincorporate this back into the final models built; had to remove because the local function was making parallelization fail
                          # oo.StatelessColumnTransformer(columns=['capital-gain', 'capital-loss'],
                          #                               custom_function=column_log),
                          oo.CenterScaleTransformer(),
                          oo.DummyEncodeTransformer(oo.CategoricalEncoding.ONE_HOT)]

# get the expected columns at the time we do the training, based on the transformations
columns = oo.TransformerPipeline.get_expected_columns(transformations=global_transformations,
                                                      data=transformed_dataset.drop(columns=[target_variable]))

# define the models and hyper-parameters that we want to search through
infos = [
         # oo.ModelInfo(description='dummy_stratified',
         #              model=oo.DummyClassifier(oo.DummyClassifierStrategy.STRATIFIED),
         #              transformations=None, hyper_params=None, hyper_params_grid=None),
         # oo.ModelInfo(description='dummy_frequent',
         #              model=oo.DummyClassifier(oo.DummyClassifierStrategy.MOST_FREQUENT),
         #              transformations=None, hyper_params=None, hyper_params_grid=None),
         # oo.ModelInfo(description='Logistic Regression',
         #              model=oo.LogisticClassifier(),
         #              transformations=None,#[oo.RemoveCorrelationsTransformer()],
         #              hyper_params=oo.LogisticClassifierHP(),
         #              hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
         #                  penalty=['l1', 'l2'],
         #                  regularization_inverse=[0.001, 0.01, 0.1, 1, 100, 1000]))),
         # oo.ModelInfo(description='SVM Linear',
         #              model=oo.SvmLinearClassifier(), transformations=None,
         #              hyper_params=oo.SvmLinearClassifierHP(),
         #              hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
         #                  penalty=['l2'],
         #                  penalty_c=[0.001, 0.01, 0.1, 1, 100, 1000]))),
         # oo.ModelInfo(description='SVM Poly',
         #              model=oo.SvmPolynomialClassifier(),
         #              transformations=None,
         #              hyper_params=oo.SvmPolynomialClassifierHP(),
         #              hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
         #                  degree=[2, 3],
         #                  coef0=[0, 1, 10],
         #                  penalty_c=[0.001, 0.1, 100, 1000]))),
         # oo.ModelInfo(description='SVM Linear (class weights)',
         #              model=oo.SvmLinearClassifier(class_weights={negative_class: 0.3,
         #                                                          positive_class: 0.7}),
         #              transformations=None,
         #              hyper_params=oo.SvmLinearClassifierHP(),
         #              hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
         #                  penalty=['l2'],
         #                  penalty_c=[0.001, 0.01, 0.1, 1, 100, 1000]))),
         # oo.ModelInfo(description='SVM Poly (class weights)',
         #              model=oo.SvmPolynomialClassifier(class_weights={negative_class: 0.3,
         #                                                              positive_class: 0.7}),
         #              transformations=None,
         #              hyper_params=oo.SvmPolynomialClassifierHP(),
         #              hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
         #                  degree=[2, 3],
         #                  coef0=[0, 1, 10],
         #                  penalty_c=[0.001, 0.1, 100, 1000]))),
         oo.ModelInfo(description='Random Forest',
                      model=oo.RandomForestClassifier(),
                      transformations=None,
                      hyper_params=oo.RandomForestHP(),
                      hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
                          criterion='gini',
                          max_features=[int(round(len(columns) ** (1 / 2.0))),
                                        int(round(len(columns) / 2)),
                                        len(columns) - 1],
                          n_estimators=[500, 1000],
                          min_samples_leaf=[1, 50, 100]))),
         oo.ModelInfo(description='Adaboost',
                      model=oo.AdaBoostClassifier(),
                      transformations=None,
                      hyper_params=oo.AdaBoostClassifierHP(),
                      hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
                          max_depth=[3, 10, 30],
                          n_estimators=[100, 500, 1000],
                          learning_rate=[0.1, 0.5, 1]))),
         oo.ModelInfo(description='XGBoost',
                      model=oo.XGBoostClassifier(),
                      transformations=None,
                      hyper_params=oo.XGBoostTreeHP(objective=oo.XGBObjective.BINARY_LOGISTIC),
                      hyper_params_grid=oo.HyperParamsGrid(
                          params_dict=dict(colsample_bytree=[0.4, 0.7, 1.0],
                                           subsample=[0.5, 0.75, 1.0],
                                           max_depth=[3, 6, 9])))]

# infos[2].hyper_params_grid.params_grid
# infos[3].hyper_params_grid.params_grid
# infos[4].hyper_params_grid.params_grid
# infos[5].hyper_params_grid.params_grid
# infos[6].hyper_params_grid.params_grid
# infos[7].hyper_params_grid.params_grid
# infos[8].hyper_params_grid.params_grid
# infos[9].hyper_params_grid.params_grid

score_list = [oo.FBetaScore(converter=oo.TwoClassThresholdConverter(threshold=0.5,
                                                                    positive_class=positive_class),
                            beta=0.5),
              oo.AucRocScore(positive_class=positive_class),
              oo.SensitivityScore(converter=oo.TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=positive_class)),
              oo.PositivePredictiveValueScore(converter=oo.TwoClassThresholdConverter(threshold=0.5,
                                                                                      positive_class=positive_class))]
cache_directory = directory + 'searcher_fbeta_v1'

print()
print(cache_directory)
import multiprocessing
# multiprocessing.cpu_count()
searcher = oo.ModelSearcher(global_transformations=global_transformations,
                            model_infos=infos,
                            splitter=oo.ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                            resampler_function=lambda m, mt: oo.RepeatedCrossValidationResampler(
                                model=m,
                                transformations=mt,
                                scores=score_list,
                                folds=5,
                                repeats=5),
                            persistence_manager=oo.LocalCacheManager(cache_directory=cache_directory),
                            parallelization_cores=multiprocessing.cpu_count()-2)
searcher._parallelization_cores
searcher.search(data=transformed_dataset, target_variable='income')

searcher.results.model_descriptions

# searcher.results.model_names

searcher.results.plot_holdout_scores()
searcher.results.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
searcher.results.plot_resampled_scores(metric=oo.Metric.AUC_ROC)


xgb_tuner = searcher.results.tuner_results[2]
xgb_tuner.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
xgb_tuner.plot_resampled_scores(metric=oo.Metric.AUC_ROC)
xgb_tuner.plot_hyper_params_profile(metric=oo.Metric.AUC_ROC,
                                    x_axis='max_depth',
                                    line='subsample',
                                    grid='colsample_bytree')
