import features
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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
    print('\n'.join(table))


def multinomial_naive_bayes(tr_x, tr_y, te_x):
    mnb = MultinomialNB()
    y_pred = generic_classifier(mnb, tr_x, tr_y, te_x)

    return y_pred


def regularized_logistic_regression(tr_x, tr_y, te_x):
    rlr = LogisticRegression()
    y_pred = generic_classifier(rlr, tr_x, tr_y, te_x)

    return y_pred


def classification_trees(tr_x, tr_y, te_x):
    ct = DecisionTreeClassifier()
    y_pred = generic_classifier(ct, tr_x, tr_y, te_x)

    return y_pred


def random_forests(tr_x, tr_y, te_x):
    rf = RandomForestClassifier()
    y_pred = generic_classifier(rf, tr_x, tr_y, te_x)

    return y_pred


def generic_classifier(classifier, tr_x, tr_y, te_x):
    # All classifiers use same method names, so we can just call these functions generically
    classifier = classifier.fit(tr_x, tr_y)
    y_pred = classifier.predict(te_x)

    return y_pred


print("Without bi_grams:")
main(False)
print("\nWith bi_grams:")
main(True)
