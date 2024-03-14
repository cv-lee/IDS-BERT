from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def load_txt(txt_dir, strip_string='\n'):
    f = open(txt_dir, 'r')
    lines = f.readlines()
    lines = [s.strip(strip_string) for s in lines]
    f.close()
    return lines


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
    #auc = roc_auc_score(labels, preds)
    #  'auroc': auc
