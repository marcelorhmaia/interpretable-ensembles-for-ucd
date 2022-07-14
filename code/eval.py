# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


import sys
from math import sqrt

import numpy as np
import pandas
from numpy import full
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from ud_forest import DFERandomForestClassifier
from ud_bagging import UDBaggingClassifier


def read_data(file):
    data_frame = pandas.read_csv(file)
    for col in data_frame:
        if col.startswith('cpi_'):
            data_frame[col] = data_frame[col].div(1000)
    return data_frame


def cross_validation(model, dataset, class_col, data_frame, n_estimators, random_seed):
    first_feat = 1
    last_feat = 9096 if dataset.startswith('SE') else data_frame.shape[1] - 2

    uncertain_features = full((last_feat + 1 - first_feat,), False)
    for col in data_frame:
        if col != 'entrez' and col != 'class' and col != 'CID' and not col.startswith('se_'):
            uncertain_features[data_frame.columns.get_loc(col) - first_feat] = True

    precision_neg = []
    recall_neg = []
    f1score_neg = []
    support_neg = []
    precision_pos = []
    recall_pos = []
    f1score_pos = []
    support_pos = []
    accuracy = []
    balanced_accuracy = []
    roc_auc = []
    g_mean = []

    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    i = 0
    for trainIndices, testIndices in kf.split(data_frame):
        train_x = data_frame.iloc[trainIndices, first_feat:last_feat+1]
        train_y = data_frame.iloc[trainIndices].loc[:, class_col]

        test_x = data_frame.iloc[testIndices, first_feat:last_feat+1]
        test_y = data_frame.iloc[testIndices].loc[:, class_col]

        if model == 'ENB-NV':
            classifier = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators,
                                           max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                           random_state=random_seed)
        elif model == 'ENB-NV+BB':
            classifier = UDBaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_bootstrap=True)
        elif model == 'ENB-NV+BRS':
            classifier = UDBaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_subspaces=True)
        elif model == 'ENB-NV+BB+BRS':
            classifier = UDBaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_bootstrap=True, biased_subspaces=True)
        elif model == 'ENB-EV':
            classifier = BaggingClassifier(base_estimator=BernoulliNB(binarize=0.5),
                                           n_estimators=n_estimators,
                                           max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                           random_state=random_seed)
        elif model == 'ENB-EV+BB':
            classifier = UDBaggingClassifier(base_estimator=BernoulliNB(binarize=0.5), n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_bootstrap=True)
        elif model == 'ENB-EV+BRS':
            classifier = UDBaggingClassifier(base_estimator=BernoulliNB(binarize=0.5),
                                             n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_subspaces=True)
        elif model == 'ENB-EV+BB+BRS':
            classifier = UDBaggingClassifier(base_estimator=BernoulliNB(binarize=0.5), n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_bootstrap=True, biased_subspaces=True)
        elif model == 'RF-DFE':
            classifier = DFERandomForestClassifier(n_estimators=n_estimators, random_state=random_seed,
                                                   class_weight='balanced_subsample',
                                                   uncertain_features=uncertain_features)
        elif model == 'RF-DFE+BB':
            classifier = DFERandomForestClassifier(n_estimators=n_estimators, random_state=random_seed,
                                                   class_weight='balanced_subsample',
                                                   uncertain_features=uncertain_features, biased_bootstrap=True)
        elif model == 'RF-DFE+BS':
            classifier = DFERandomForestClassifier(n_estimators=n_estimators, random_state=random_seed,
                                                   class_weight='balanced_subsample',
                                                   uncertain_features=uncertain_features, biased_splitting=True)
        else:   # model == 'RF-DFE+BB+BS'
            classifier = DFERandomForestClassifier(n_estimators=n_estimators, random_state=random_seed,
                                                   class_weight='balanced_subsample',
                                                   uncertain_features=uncertain_features, biased_bootstrap=True,
                                                   biased_splitting=True)

        classifier.fit(train_x, train_y)

        proba = classifier.predict_proba(test_x)
        pred_y = classifier.classes_.take(np.argmax(proba, axis=1), axis=0)

        report = classification_report(test_y, pred_y, target_names=['neg', 'pos'], output_dict=True,
                                       zero_division=0)

        precision_neg.append(report['neg']['precision'])
        recall_neg.append(report['neg']['recall'])
        f1score_neg.append(report['neg']['f1-score'])
        support_neg.append(report['neg']['support'])
        precision_pos.append(report['pos']['precision'])
        recall_pos.append(report['pos']['recall'])
        f1score_pos.append(report['pos']['f1-score'])
        support_pos.append(report['pos']['support'])
        accuracy.append(report['accuracy'])
        balanced_accuracy.append(balanced_accuracy_score(test_y, pred_y))
        roc_auc.append(roc_auc_score(test_y, proba[:, 1]))
        g_mean.append(sqrt(report['neg']['recall'] * report['pos']['recall']))

        i += 1

    results = pandas.DataFrame(data={'precision(neg)': precision_neg, 'recall(neg)': recall_neg,
                                     'f1-score(neg)': f1score_neg, 'support(neg)': support_neg,
                                     'precision(pos)': precision_pos, 'recall(pos)': recall_pos,
                                     'f1-score(pos)': f1score_pos, 'support(pos)': support_pos,
                                     'accuracy': accuracy, 'b-accuracy': balanced_accuracy,
                                     'roc-auc': roc_auc, 'g-mean': g_mean})
    results.loc['mean'] = results.mean()
    results['f1-score(neg)']['mean'] = 2 * results['precision(neg)']['mean'] * results['recall(neg)']['mean']
    if results['f1-score(neg)']['mean'] != 0:
        results['f1-score(neg)']['mean'] /= results['precision(neg)']['mean'] + results['recall(neg)']['mean']
    results['f1-score(pos)']['mean'] = 2 * results['precision(pos)']['mean'] * results['recall(pos)']['mean']
    if results['f1-score(pos)']['mean'] != 0:
        results['f1-score(pos)']['mean'] /= results['precision(pos)']['mean'] + results['recall(pos)']['mean']
    results['f1-score(pos)']['std'] = np.nan
    results['g-mean']['mean'] = sqrt(results['recall(neg)']['mean'] * results['recall(pos)']['mean'])
    print(results)


models = {'ENB-NV', 'ENB-NV+BB', 'ENB-NV+BRS', 'ENB-NV+BB+BRS',
          'ENB-EV', 'ENB-EV+BB', 'ENB-EV+BRS', 'ENB-EV+BB+BRS',
          'RF-DFE', 'RF-DFE+BB', 'RF-DFE+BS', 'RF-DFE+BB+BS'}

datasets = {'AG-Worm', 'AG-Fly', 'AG-Mouse', 'AG-Yeast',
            'SE-Nausea', 'SE-Headache', 'SE-Dermatitis', 'SE-Rash', 'SE-Vomiting', 'SE-Dizziness'}

se_cols = {'SE-Nausea': 'se_C0027497',
           'SE-Headache': 'se_C0018681',
           'SE-Dermatitis': 'se_C0011603',
           'SE-Rash': 'se_C0015230',
           'SE-Vomiting': 'se_C0042963',
           'SE-Dizziness': 'se_C0012833'}

if len(sys.argv) < 3:
    print('Usage: python eval.py <model> <dataset>')
elif sys.argv[1] not in models:
    print(sys.argv[1], 'is not a valid model.')
    print('Valid models are:', models)
elif sys.argv[2] not in datasets:
    print(sys.argv[2], 'is not a valid dataset.')
    print('Valid datasets are:', datasets)
else:
    model = sys.argv[1]
    dataset = sys.argv[2]

    pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', 170)
    datasets_dir = '../data/'
    estimators = 500
    seed = 1 if dataset.startswith('SE') else 2
    class_col = se_cols[dataset] if dataset.startswith('SE') else 'class'
    filename = 'SE' if dataset.startswith('SE') else dataset
    filename += '.csv'

    print('dataset =', dataset, '| model =', model)
    df = read_data(datasets_dir + filename)
    cross_validation(model, dataset, class_col, df, estimators, seed)
