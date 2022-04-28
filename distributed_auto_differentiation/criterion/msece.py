import torch.nn as nn

def msecse(x, y, mse_inds=[0], clf_inds=[1,2], alpha=1.):
    x_mse = x[:, mse_inds]
    y_mse = y[:, mse_inds]
    x_clf = x[:, clf_inds]
    y_clf = y[:, clf_inds]
    mse_loss = nn.MSELoss()(x_mse, y_mse)
    clf_loss = nn.CrossEntropyLoss()(x_clf, y_clf)
    return alpha*mse_loss + (1- alpha)* clf_loss
