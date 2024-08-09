import numpy as np
import torch


def compute_metrics(eval_preds):
    acc = 0
    scores_list = eval_preds.predictions[0]
    for i in range(len(scores_list[0])):
        for j in range(6):
            acc += int(scores_list[j][i] > scores_list[j + 1][i])
    result = {}
    result["accuracy"] = float(acc) / 6 / len(scores_list[0])

    result["mean_score"] = np.mean(scores_list)
    result["mean_std"] = np.std(scores_list)
    result["mean_score_sum"] = np.sum(scores_list, axis=0).mean()

    return result


def reward_loss_7wise(scores):
    loss = 0
    for j in range(7):
        log_denominator = torch.logsumexp(scores[j:], dim=0)
        loss += -(scores[j] - log_denominator)
    return loss
