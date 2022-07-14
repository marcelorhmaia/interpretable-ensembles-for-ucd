# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


import sys

import numpy as np
import pandas
from numpy import full

from ud_naive_bayes import InterpretableBernoulliNB
from ud_bagging import UDBaggingClassifier
from ud_forest import DFERandomForestClassifier


def read_data(file):
    return pandas.read_csv(file)


def top_features(model, dataset, data_frame, n_estimators, random_seed):
    first_feat = 1;
    last_feat = data_frame.shape[1] - 2

    uncertain_features = full((last_feat + 1 - first_feat,), False)
    for col in data_frame:
        if col != 'entrez' and col != 'class':
            uncertain_features[data_frame.columns.get_loc(col) - first_feat] = True

    feat_importances = pandas.DataFrame(data={'name': data_frame.columns[first_feat:-1]})

    train_x = data_frame.iloc[:, first_feat:-1]
    train_y = data_frame.loc[:, 'class']

    if model == 'ENB-EV+BRS':
        classifier = UDBaggingClassifier(base_estimator=InterpretableBernoulliNB(binarize=0.5),
                                         n_estimators=n_estimators,
                                         max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                         random_state=random_seed, uncertain_features=uncertain_features,
                                         biased_subspaces=True)
    else:   # model == 'RF-DFE+BB+BS'
        classifier = DFERandomForestClassifier(n_estimators=n_estimators, random_state=random_seed,
                                               class_weight='balanced_subsample',
                                               uncertain_features=uncertain_features, biased_bootstrap=True,
                                               biased_splitting=True)

    classifier.fit(train_x, train_y)

    feat_importances['importance'] = classifier.feature_importances_
    feat_importances['rank'] = feat_importances['importance'].rank(method='min', ascending=False)

    print('top 10 features', '(conditional probabilities)' if model.startswith('ENB') else '')
    print(feat_importances.sort_values(by=['rank']).head(10))

    if model.startswith('ENB'):
        feat_importances['importance'] = classifier.sufficiency_based_feature_importances(train_x)
        feat_importances['rank'] = feat_importances['importance'].rank(method='min', ascending=False)
        print()
        print('top 10 features (minimal sufficient sets)')
        print(feat_importances.sort_values(by=['rank']).head(10))


models = {'ENB-EV+BRS', 'RF-DFE+BB+BS'}

datasets = {'AG-Worm', 'AG-Fly', 'AG-Mouse', 'AG-Yeast'}

if len(sys.argv) < 3:
    print('Usage: python eval_top_features.py <model> <dataset>')
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
    seed = 2
    filename = dataset + '.csv'

    print('dataset =', dataset, '| model =', model)
    df = read_data(datasets_dir + filename)
    top_features(model, dataset, df, estimators, seed)
