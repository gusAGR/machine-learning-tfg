from sklearn.metrics import make_scorer, fbeta_score, recall_score


def accuracy_precision_recall_specifity_f2_score():
    """

    :return:
    """
    f2_score = make_scorer(fbeta_score, beta=2)
    specifity = make_scorer(recall_score, pos_label=0)
    metrics = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'specifity': specifity, 'f2_score': f2_score }
    return metrics
