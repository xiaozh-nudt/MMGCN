def statistic(TP,TN,P,N):
    FP = N - TN
    FN = P - TP
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP+ FP + TN + FN)
    except BaseException:
        precision = 0
        recall = 0
        accuracy = 0
    if precision+recall != 0:
        F1 = precision*recall/2*(precision+recall)
    else:
        F1 = -1
    result ={
        'TP':TP,
        'TN':TN,
        'FP':FP,
        'FN':FN,
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'F1 score':F1
    }
    return result