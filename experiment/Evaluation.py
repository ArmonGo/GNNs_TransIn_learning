
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_figures(rst, y_true):
    m_ls = []
    precision_ls = []
    recall_ls = []
    fscore_ls = []
    roc_auc = []
    pr_auc = []
    repeat = []
    for cnt in rst.keys():
        for m, (y_pred , y_mask) in rst[cnt].items():
            if len(y_true.shape) >1:
                y_pred_class = (y_pred > 0.5).int()
            elif y_true.max()>1:
                y_pred_class = y_pred.max(dim=1)[1]
            else:
                y_pred_class = (y_pred > 0.5).int()
            m_ls.append(m)
            y_p, y_t, y_c = y_pred[y_mask], y_true[y_mask], y_pred_class[y_mask]
            if len(y_true.shape) >1:
                precision, recall, fscore, support = precision_recall_fscore_support(y_t, y_c, average='weighted')
                roc_auc_s = roc_auc_score(y_t, y_p, average='weighted')
                pr_auc_s = None
            elif y_true.max() == 1:
                precision, recall, fscore, support = precision_recall_fscore_support(y_t, y_c, average='binary')
                roc_auc_s = roc_auc_score(y_t, y_p)
                pr, re, _ = precision_recall_curve(y_t, y_p)
                pr_auc_s = auc(re, pr)
            else:
                precision, recall, fscore, support = precision_recall_fscore_support(y_t, y_c, average='weighted')
                roc_auc_s = roc_auc_score(y_t, y_p, average='weighted', multi_class ='ovr')
                pr_auc_s = None
            repeat.append(cnt)
            precision_ls.append(precision)
            recall_ls.append(recall)
            fscore_ls.append(fscore)
            # support_ls.append(support)
            roc_auc.append(roc_auc_s)
            pr_auc.append(pr_auc_s)
    scores_tb = pd.DataFrame( {
                            'repeat': repeat, 
                            'model': m_ls, 
                            'precision': precision_ls, 
                            'recall': recall_ls,
                            'F1 score': fscore_ls,
                            'AUROC' : roc_auc,
                            'PRAUC': pr_auc} )
    return scores_tb


def plot_diff(df, title, save = './/plot//', metrics = ['precision', 'recall', 'F1 score', 'AUROC', 'PRAUC'], figure_size =(5, 10), legendsize=15 ):
    # Set up the subplots with 5 rows and 1 column
    fig, axes = plt.subplots(len(metrics), 1, figsize=figure_size, sharex=True)

    # Create the subplots with the vertical layout
    for i, metric in enumerate(metrics):
        sns.lineplot(
            x='model', y=metric, hue='type', data=df, ax=axes[i], style="type", markers=True
        )
        # axes[i].set_title(metric)
        axes[i].set_ylabel(metric, fontsize=17)
        # Only set x-axis label on the bottom plot
        if i < len(metrics) - 1:
            axes[i].set_xlabel('')
        if i != len(metrics)-1:
            axes[i].get_legend().remove()
        else:
            handles, labels = axes[i].get_legend_handles_labels()
            hl = list(zip(handles, labels))
            hl.sort(key=lambda t: ['transductive', 'inductive (weak)', 'inductive (medium)', 'inductive (strong)'].index(t[1]))
            handles2, labels2 = zip(*hl)
            axes[i].get_legend().remove()

    fig.legend(handles=handles2, labels=labels2, loc='lower center', ncol=4, fontsize = legendsize)
    # Adjust layout to avoid overlap and ensure clarity
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.95/figure_size[1])
    if save is not None:
        plt.savefig(save + title + '.png')
    plt.show()
    