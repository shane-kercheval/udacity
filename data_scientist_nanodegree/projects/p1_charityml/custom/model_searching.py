import copy
from typing import Union

import oolearning as oo
import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score

from projects.p1_charityml.custom.helpers import column_log

pd.set_option('display.width', 500)

##############################################################################################################
class BinaryAucRocScore(oo.AucRocScore):
    """
    Calculates the AUC of the ROC curve as defined by sklearn's `roc_auc_score()`
        http://scikit-learn.org/stable/modules/generated/sklearn.score_names.roc_auc_score.html
    """
    def __init__(self, positive_class, threshold: float=0.5):
        super().__init__(positive_class=positive_class)
        self._threshold = threshold

    @property
    def name(self) -> str:
        return 'BINARY_AUC'

    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:

        return roc_auc_score(y_true=[1 if x == self._positive_class else 0 for x in actual_values],
                             # binary makes it so it converts the "scores" to predictions
                             y_score=[1 if x > self._threshold else 0 for x in predicted_values[self._positive_class].values])
##############################################################################################################
print(os.getcwd())
directory = 'projects/p1_charityml/'
csv_file = os.path.join(directory, 'census.csv')
target_variable = 'income'

os.path.isfile(csv_file)

explore = oo.ExploreClassificationDataset.from_csv(csv_file_path=csv_file,
                                                   target_variable=target_variable)
negative_class = '<=50K'
positive_class = '>50K'

explore.dataset.head(20)


###################
pipeline = oo.TransformerPipeline([oo.CenterScaleTransformer(),
                                   oo.DummyEncodeTransformer(oo.CategoricalEncoding.ONE_HOT)])
pca_data = pipeline.fit_transform(explore.dataset.drop(columns=target_variable))
pca_data.shape

pca_transformer = oo.PCATransformer(percent_variance_explained=None)
pca_transformer.fit(data_x=pca_data)

pca_transformer.cumulative_explained_variance
pca_transformer.number_of_components
pca_transformer.state

pca_transformer.plot_cumulative_variance()





#######################
#
global_transformations = [oo.ImputationTransformer(),
                          oo.StatelessColumnTransformer(columns=['capital-gain', 'capital-loss'],
                                                        custom_function=column_log),
                          oo.CenterScaleTransformer(),
                          oo.DummyEncodeTransformer(oo.CategoricalEncoding.ONE_HOT)]
#
# get the expected columns at the time we do the training, based on the transformations
columns = oo.TransformerPipeline.get_expected_columns(transformations=global_transformations,
                                                      data=explore.dataset.drop(columns=[target_variable]))

# define the models and hyper-parameters that we want to search through
infos = [
         oo.ModelInfo(description='dummy_stratified',
                      model=oo.DummyClassifier(oo.DummyClassifierStrategy.STRATIFIED),
                      transformations=None, hyper_params=None, hyper_params_grid=None),
         oo.ModelInfo(description='dummy_frequent',
                      model=oo.DummyClassifier(oo.DummyClassifierStrategy.MOST_FREQUENT),
                      transformations=None, hyper_params=None, hyper_params_grid=None),
         oo.ModelInfo(description='Logistic Regression',
                      model=oo.LogisticClassifier(),
                      transformations=None,#[oo.RemoveCorrelationsTransformer()],
                      hyper_params=oo.LogisticClassifierHP(),
                      hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
                          penalty=['l1', 'l2'],
                          regularization_inverse=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 100]))),
         oo.ModelInfo(description='SVM Linear',
                      model=oo.SvmLinearClassifier(), transformations=None,
                      hyper_params=oo.SvmLinearClassifierHP(),
                      hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
                          penalty=['l2'],
                          penalty_c=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 100, 1000]))),
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
                                        25,
                                        int(round(len(columns) / 2)),
                                        75,
                                        len(columns) - 1],
                          n_estimators=[100, 250, 500, 1000],
                          min_samples_leaf=[1, 25, 50, 75, 100]))),
         oo.ModelInfo(description='Adaboost',
                      model=oo.AdaBoostClassifier(),
                      transformations=None,
                      hyper_params=oo.AdaBoostClassifierHP(),
                      hyper_params_grid=oo.HyperParamsGrid(params_dict=dict(
                          max_depth=[2, 3, 5, 10, 30],
                          n_estimators=[50, 100, 150, 500],
                          learning_rate=[0.01, 0.05, 0.1, 0.5]))),
         oo.ModelInfo(description='XGBoost',
                      model=oo.XGBoostClassifier(),
                      transformations=None,
                      hyper_params=oo.XGBoostTreeHP(objective=oo.XGBObjective.BINARY_LOGISTIC),
                      hyper_params_grid=oo.HyperParamsGrid(
                          params_dict=dict(colsample_bytree=[0.1, 0.25, 0.4, 0.7],
                                           subsample=[0.5, 0.75, 1.0],
                                           max_depth=[6, 9, 15, 20])))]

# infos[2].hyper_params_grid.params_grid
# infos[3].hyper_params_grid.params_grid
# infos[4].hyper_params_grid.params_grid
# infos[5].hyper_params_grid.params_grid
# infos[6].hyper_params_grid.params_grid
# infos[7].hyper_params_grid.params_grid
# infos[8].hyper_params_grid.params_grid
# infos[9].hyper_params_grid.params_grid

score_list = [BinaryAucRocScore(positive_class=positive_class, threshold=0.5),
              oo.AucRocScore(positive_class=positive_class),
              oo.FBetaScore(converter=oo.TwoClassThresholdConverter(threshold=0.5,
                                                                    positive_class=positive_class),
                            beta=0.5),
              oo.SensitivityScore(converter=oo.TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=positive_class)),
              oo.PositivePredictiveValueScore(converter=oo.TwoClassThresholdConverter(threshold=0.5,
                                                                                      positive_class=positive_class))]
model_cache_directory = directory + 'searcher_fbeta_v1'
resampler_cache_directory = directory + 'searcher_fbeta_v1_resample_cache'

print()
print(model_cache_directory)
import multiprocessing
# multiprocessing.cpu_count()
searcher = oo.ModelSearcher(global_transformations=[t.clone() for t in global_transformations],
                            model_infos=infos,
                            splitter=oo.ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                            resampler_function=lambda m, mt: oo.RepeatedCrossValidationResampler(
                                model=m,
                                transformations=mt,
                                scores=[s.clone() for s in score_list],
                                folds=5,
                                repeats=5),
                            model_persistence_manager=oo.LocalCacheManager(cache_directory=model_cache_directory),
                            resampler_persistence_manager=oo.LocalCacheManager(cache_directory=resampler_cache_directory),
                            #parallelization_cores=0)
                            parallelization_cores=multiprocessing.cpu_count()-2)
searcher._parallelization_cores
searcher.search(data=explore.dataset, target_variable='income')

searcher.results.model_descriptions

# searcher.results.model_names

searcher.results.plot_resampled_scores(metric=oo.Metric.AUC_ROC)
searcher.results.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
searcher.results.plot_holdout_scores()

temp = searcher.results.best_tuned_results

log_tuner = searcher.results.tuner_results[2]
log_tuner.plot_resampled_scores(metric=oo.Metric.AUC_ROC)
log_tuner.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
log_tuner.plot_hyper_params_profile(metric=oo.Metric.AUC_ROC,
                                    x_axis='regularization_inverse',
                                    line='penalty')

svm_tuner = searcher.results.tuner_results[3]
svm_tuner.plot_resampled_scores(metric=oo.Metric.AUC_ROC)
svm_tuner.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
svm_tuner.plot_hyper_params_profile(metric=oo.Metric.AUC_ROC,
                                    x_axis='max_depth',
                                    line='n_estimators',
                                    grid='learning_rate')

rf_tuner = searcher.results.tuner_results[4]
rf_tuner.plot_resampled_scores(metric=oo.Metric.AUC_ROC)
rf_tuner.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
rf_tuner.plot_hyper_params_profile(metric=oo.Metric.AUC_ROC,
                                    x_axis='max_features',
                                    line='min_samples_leaf',
                                    grid='n_estimators')

ada_tuner = searcher.results.tuner_results[5]
ada_tuner.plot_resampled_scores(metric=oo.Metric.AUC_ROC)
ada_tuner.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
ada_tuner.plot_hyper_params_profile(metric=oo.Metric.AUC_ROC,
                                    x_axis='max_depth',
                                    line='learning_rate',
                                    grid='n_estimators')

xgb_tuner = searcher.results.tuner_results[6]
temp = xgb_tuner._tune_results_objects.iloc[0].resampler_object.resampled_scores
temp.AUC_ROC.mean()
temp.AUC_ROC.std()
xgb_tuner.resampled_stats
xgb_tuner.plot_resampled_scores(metric=oo.Metric.AUC_ROC)
xgb_tuner.plot_resampled_scores(metric=oo.Metric.FBETA_SCORE)
xgb_tuner.plot_hyper_params_profile(metric=oo.Metric.AUC_ROC,
                                    x_axis='max_depth',
                                    line='subsample',
                                    grid='colsample_bytree')
##############################################################################################################
# Try Model Stacker
##############################################################################################################
model_infos = [oo.ModelInfo(description='Logistic Regression',
                            model=oo.LogisticClassifier(),
                            hyper_params=oo.LogisticClassifierHP(penalty=,
                                                                 regularization_inverse=,
                                                                 ),
                            transformations=None),
               oo.ModelInfo(description='SVM',
                            model=oo.SvmLinearClassifier(),
                            hyper_params=oo.SvmLinearClassifierHP(penalty=,
                                                                  penalty_c=,
                                                                  ),
                            transformations=None),
               oo.ModelInfo(description='Random Forest',
                            model=oo.RandomForestClassifier(),
                            hyper_params=oo.RandomForestHP(criterion='gini',
                                                           num_features=,
                                                           max_features=,
                                                           n_estimators=,
                                                           max_depth=,
                                                           max_leaf_nodes=,
                                                           ),
                            transformations=None),
               oo.ModelInfo(description='AdaBoost',
                            model=oo.AdaBoostClassifier(),
                            hyper_params=oo.AdaBoostClassifierHP(max_depth=,
                                                                 n_estimators=,
                                                                 learning_rate=,
                                                                 ),
                            transformations=None),
               oo.ModelInfo(description='XGBoost',
                            model=oo.XGBoostClassifier(),
                            hyper_params=oo.XGBoostTreeHP(objective=oo.XGBObjective.BINARY_LOGISTIC,
                                                          max_depth=, learning_rate=,
                                                          n_estimators=,
                                                          colsample_bytree=,
                                                          subsample=,
                                                          ),
                            transformations=None)]

converter = oo.TwoClassThresholdConverter(threshold=0.5,positive_class=positive_class)
model_stacker = oo.ModelStacker(base_models=model_infos,
                                scores=[oo.AucRocScore(positive_class=positive_class),
                                        oo.FBetaScore(beta=0.5, converter=copy.deepcopy(converter))],
                                stacking_model=oo.XGBoostClassifier(),
                                stacking_transformations=None)

trainer_stacker = oo.ModelTrainer(model=model_stacker,
                                  model_transformations=[t.clone() for t in global_transformations],
                                  splitter=oo.ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                  evaluator=oo.TwoClassProbabilityEvaluator(converter=copy.deepcopy(converter)))

trainer_stacker.train(data=explore.dataset, target_variable='expenses', hyper_params=)
trainer_stacker.holdout_evaluator.all_quality_metrics

##############################################################################################################
# Build Final Model
##############################################################################################################
final_model = oo.XGBoostClassifier()
final_hyper_params = oo.XGBoostTreeHP(
                                      objective=oo.XGBObjective.BINARY_LOGISTIC,
                                      colsample_bytree=0.4,
                                      subsample=1.0,
                                      max_depth=9,
                                      )


#final_converter_converter = copy.deepcopy(converter)
final_converter = oo.TwoClassThresholdConverter(threshold=0.25,positive_class=positive_class)
#splitter = oo.ClassificationStratifiedDataSplitter(holdout_ratio=0.20)
splitter = None  # train on all the data
evaluator = oo.TwoClassProbabilityEvaluator(converter=final_converter)
trainer = oo.ModelTrainer(model=final_model,
                          model_transformations=
                            # in the final data, there are values like ' value' in
                            # categorical columns; need to do this before we do any other
                            # transforamtions
                            [oo.StatelessColumnTransformer(columns=explore.categoric_features,
                                                           custom_function=lambda x: x.str.strip())] +
                            [g.clone() for g in global_transformations],
                          splitter=splitter,
                          evaluator=evaluator,
                          scores=[BinaryAucRocScore(positive_class=positive_class,
                                                    threshold=0.25)])
# train on entire original dataset
trainer.train(data=explore.dataset, target_variable=target_variable, hyper_params=final_hyper_params)

trainer.training_evaluator.all_quality_metrics
trainer.holdout_evaluator.all_quality_metrics

trainer.training_scores[0].value
trainer.holdout_scores[0].value

trainer.holdout_evaluator.plot_roc_curve()
trainer.holdout_evaluator.plot_precision_recall_curve()

##############################################################################################################
# Submission
##############################################################################################################
csv_file = directory+'test_census.csv'
test_dataset = pd.read_csv(csv_file)
test_dataset.shape

# test_dataset['workclass'].unique()
# [(test_dataset[x].unique(), explore.dataset[x].unique()) for x in explore.categoric_features]

indexes = test_dataset['Unnamed: 0']

predictions = trainer.predict(test_dataset.drop(columns='Unnamed: 0'))
#test_converter = copy.deepcopy(converter)
test_converter = oo.TwoClassThresholdConverter(threshold=0.25,positive_class=positive_class)
class_predictions = test_converter.convert(predictions)
income_value = [0 if x == '<=50K' else 1 for x in class_predictions]
pd.DataFrame({'id': indexes, 'income': income_value}).to_csv(directory + 'submission_3_all_data.csv',
                                                             index=False)

