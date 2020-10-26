import features
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def main(with_bi_grams):
    all_data = features.add_features(with_bi_grams)
    tr_x, tr_y, te_x, te_y = split_data(all_data)

    mnb_y = multinomial_naive_bayes(tr_x, tr_y, te_x)
    mnb_perf = performance(te_y, mnb_y, "MNB")

    rlr_y = regularized_logistic_regression(tr_x, tr_y, te_x)
    rlr_perf = performance(te_y, rlr_y, "RLR")

    ct_y = classification_trees(tr_x, tr_y, te_x)
    ct_perf = performance(te_y, ct_y, "CT")

    rf_y = random_forests(tr_x, tr_y, te_x)
    rf_perf = performance(te_y, rf_y, "RF")

    show_scores(mnb_perf, rlr_perf, ct_perf, rf_perf)


def split_data(data):
    training_data = data[:640, 1:]
    test_data = data[-160:, 1:]

    training_x = training_data[:, :-1]
    training_y = training_data[:, -1].astype('int')

    test_x = test_data[:, :-1]
    test_y = test_data[:, -1].astype('int')

    return training_x, training_y, test_x, test_y


def performance(y_true, y_pred, classifier):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    return [classifier, accuracy, precision, recall, f1]


def show_scores(mnb_perf, rlr_perf, ct_perf, rf_perf):
    # Stolen from stack overflow, just prints nicely
    scores = ["Classifier", "Accuracy", "Precision", "Recall", "F1"]
    matrix = [scores, mnb_perf, rlr_perf, ct_perf, rf_perf]
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print("\n====================================================================================")
    print('\n'.join(table))
    print("====================================================================================")


def multinomial_naive_bayes(tr_x, tr_y, te_x):
    mnb = MultinomialNB()

    # Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    alpha = [x for x in np.linspace(start=0, stop=1, num=5)]
    # Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    fit_prior = [True, False]
    # Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
    class_prior = [None]

    random_grid = {"alpha": alpha,
                   "fit_prior": fit_prior,
                   "class_prior": class_prior}
    y_pred = generic_classifier(mnb, tr_x, tr_y, te_x, random_grid)

    return y_pred


def regularized_logistic_regression(tr_x, tr_y, te_x):
    rlr = LogisticRegression()

    # Used to specify the norm used in the penalization
    penalty = ["l2", "none"]
    # Inverse of regularization strength
    c = [x for x in np.linspace(start=0.4, stop=1.4, num=5)]
    # Weights associated with classes in the form
    class_weight = ["balanced", None]

    random_grid = {"penalty": penalty,
                   "C": c,
                   "class_weight": class_weight}
    y_pred = generic_classifier(rlr, tr_x, tr_y, te_x, random_grid)

    return y_pred


def classification_trees(tr_x, tr_y, te_x):
    ct = DecisionTreeClassifier()

    # Number of features to consider at every split
    max_features = ['sqrt', 'log2', None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(50, 120, num=7)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Complexity parameter used for Minimal Cost-Complexity Pruning
    ccp_alpha = [x for x in np.linspace(start=0.0, stop=1.0, num=5)]

    random_grid = {'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'ccp_alpha': ccp_alpha}
    y_pred = generic_classifier(ct, tr_x, tr_y, te_x, random_grid)

    return y_pred


def random_forests(tr_x, tr_y, te_x):
    rf = RandomForestClassifier()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=1000, stop=3500, num=5)]
    # Number of features to consider at every split
    max_features = ['sqrt', 'log2', None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(50, 120, num=7)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    # Complexity parameter used for Minimal Cost-Complexity Pruning
    ccp_alpha = [x for x in np.linspace(start=0.0, stop=1.0, num=5)]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'ccp_alpha': ccp_alpha}
    y_pred = generic_classifier(rf, tr_x, tr_y, te_x, random_grid)

    return y_pred


def generic_classifier(classifier, tr_x, tr_y, te_x, random_grid):
    # All classifiers use same method names, so we can just call these functions generically
    classifier = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, n_iter=10, cv=4, verbose=0,
                                    random_state=42, n_jobs=-1)
    classifier = classifier.fit(tr_x, tr_y)
    # print(classifier.best_params_)
    best_classifier = classifier.best_estimator_

    #TODO: beste feauters vinden plzzzz
    # print(type(best_classifier))
    # if type(best_classifier) == 'sklearn.linear_model._logistic.LogisticRegression':
    #     print("YEET")
    #     print(best_classifier.coef_)

    y_pred = best_classifier.predict(te_x)

    return y_pred


print("Without bi_grams:")
main(False)
print("\nWith bi_grams:")
main(True)
