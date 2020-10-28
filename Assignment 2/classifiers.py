import features
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def main(with_bi_grams, min_uni_gram_count, min_bi_gram_count):
    all_data, column_headers = features.add_features(with_bi_grams, min_uni_gram_count, min_bi_gram_count)
    tr_x, tr_y, te_x, te_y = split_data(all_data)

    mnb_y, coef_dict_mnb = multinomial_naive_bayes(tr_x, tr_y, te_x, column_headers, "MNB")
    mnb_perf = performance(te_y, mnb_y, "MNB")

    rlr_y, coef_dict_rlr = regularized_logistic_regression(tr_x, tr_y, te_x, column_headers, "RLR")
    rlr_perf = performance(te_y, rlr_y, "RLR")

    ct_y, coef_dict_ct = classification_trees(tr_x, tr_y, te_x, column_headers, "CT")
    ct_perf = performance(te_y, ct_y, "CT")

    rf_y, coef_dict_rf = random_forests(tr_x, tr_y, te_x, column_headers, "RF")
    rf_perf = performance(te_y, rf_y, "RF")

    best_features(coef_dict_mnb, coef_dict_rlr, coef_dict_ct, coef_dict_rf)

    show_scores(mnb_perf, rlr_perf, ct_perf, rf_perf)

    return mnb_perf, rlr_perf, ct_perf, rf_perf


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


def best_features(coef_dict_mnb, coef_dict_rlr, coef_dict_ct, coef_dict_rf):
    print("MNB_features:\n", coef_dict_mnb[:5], "\n", coef_dict_mnb[-5:])
    print("RLR_features:\n", coef_dict_rlr[:5], "\n", coef_dict_rlr[-5:])
    print("CT_features:\n", coef_dict_ct[:5], "\n", coef_dict_ct[-5:])
    print("RF_features:\n", coef_dict_rf[:5], "\n", coef_dict_rf[-5:])


def multinomial_naive_bayes(tr_x, tr_y, te_x, column_headers, class_type):
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
    y_pred, coef_dict = generic_classifier(mnb, tr_x, tr_y, te_x, random_grid, column_headers, class_type)

    return y_pred, coef_dict


def regularized_logistic_regression(tr_x, tr_y, te_x, column_headers, class_type):
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
    y_pred, coef_dict = generic_classifier(rlr, tr_x, tr_y, te_x, random_grid, column_headers, class_type)

    return y_pred, coef_dict


def classification_trees(tr_x, tr_y, te_x, column_headers, class_type):
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
    y_pred, coef_dict = generic_classifier(ct, tr_x, tr_y, te_x, random_grid, column_headers, class_type)

    return y_pred, coef_dict


def random_forests(tr_x, tr_y, te_x, column_headers, class_type):
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
    bootstrap = [True, False]
    # Complexity parameter used for Minimal Cost-Complexity Pruning
    ccp_alpha = [x for x in np.linspace(start=0.0, stop=1.0, num=5)]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'ccp_alpha': ccp_alpha}
    y_pred, coef_dict = generic_classifier(rf, tr_x, tr_y, te_x, random_grid, column_headers, class_type)

    return y_pred, coef_dict


def generic_classifier(classifier, tr_x, tr_y, te_x, random_grid, column_headers, class_type):
    # All classifiers use same method names, so we can just call these functions generically
    classifier = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, n_iter=10, cv=4, verbose=0,
                                    random_state=42, n_jobs=-1)
    classifier = classifier.fit(tr_x, tr_y)
    best_classifier = classifier.best_estimator_
    coef_dict = get_coef_dict(best_classifier, column_headers, class_type)
    y_pred = best_classifier.predict(te_x)

    return y_pred, coef_dict


def get_coef_dict(best_classifier, column_headers, class_type):
    if class_type == "MNB" or class_type == "RLR":
        feature_coefs = list(zip(best_classifier.coef_[0], column_headers))
    else:
        feature_coefs = list(zip(best_classifier.feature_importances_.tolist(), column_headers))

    feature_coefs.sort(key=lambda tup: tup[0])
    most_neg = feature_coefs[0][0]
    feature_coefs = [(x - most_neg, j) for x, j in feature_coefs]
    feature_coefs_norm = [(float(i) / max(i for i, j in feature_coefs), j) for i, j in feature_coefs]

    return feature_coefs_norm


def loop_through_min_features(with_bi_grams):
    min_count_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_axis = min_count_list
    y_mnb = []
    y_rlr = []
    y_ct = []
    y_rf = []

    if with_bi_grams:
        title = "bigrams"
    else:
        title = "unigrams"

    if not os.path.exists('performance' + title + '.pkl'):
        for i in min_count_list:
            mnb_perf, rlr_perf, ct_perf, rf_perf = main(with_bi_grams, i, i)
            y_mnb.append(mnb_perf[1])  # accuracy van mnb
            y_rlr.append(rlr_perf[1])
            y_ct.append(ct_perf[1])
            y_rf.append(rf_perf[1])
        f = open('performance' + title + '.pkl', 'wb')
        pickle.dump([y_mnb, y_rlr, y_ct, y_rf, min_count_list], f)
        f.close()
    else:
        f = open('performance' + title + '.pkl', 'rb')
        y_mnb, y_rlr, y_ct, y_rf, min_count_list_pkl = pickle.load(f)
        if not min_count_list == min_count_list_pkl:
            y_mnb = []
            y_rlr = []
            y_ct = []
            y_rf = []
            for i in min_count_list:
                mnb_perf, rlr_perf, ct_perf, rf_perf = main(with_bi_grams, i, i)
                y_mnb.append(mnb_perf[1])  # accuracy van mnb
                y_rlr.append(rlr_perf[1])
                y_ct.append(ct_perf[1])
                y_rf.append(rf_perf[1])
            f = open('performance' + title + '.pkl', 'wb')
            pickle.dump([y_mnb, y_rlr, y_ct, y_rf, min_count_list], f)
        f.close()

    fig, ax = plt.subplots()
    ax.scatter(x_axis, y_mnb, c="#191e38", label="MNB")
    ax.scatter(x_axis, y_rlr, c="#0ba667", label="RLR")
    ax.scatter(x_axis, y_ct, c="#b93132", label="CT")
    ax.scatter(x_axis, y_rf, c="#ffbf5b", label="RF")

    ax.legend()
    plt.xlim(min_count_list[0]-1, min_count_list[-1]+1)
    plt.ylim(0.5, 1.0)
    plt.xticks(min_count_list)
    plt.xlabel('Min feature count')
    plt.ylabel('Accuracy')
    if with_bi_grams:
        plt.title("Unigrams and Bigrams")
    else:
        plt.title("Unigrams")

    plt.show()


print("Without bi_grams:")
loop_through_min_features(False)
print("\nWith bi_grams:")
loop_through_min_features(True)
