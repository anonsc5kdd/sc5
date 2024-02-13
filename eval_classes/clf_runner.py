from dgl.nn import GraphConv
import torch.nn.functional as F
import torch
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

import torch
import torch.nn as nn
import torch.optim as optim

class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        self.one_hot = OneHotEncoder(sparse_output=False)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        # Pass the input through the linear layer
        #output = self.dropout(self.act(self.fc(x)))
        output=self.act(self.fc(x))
        output = self.act(self.fc2(output))
        return output

class ClassifierTrainer:
    def __init__(self, input_dim, num_classes, learning_rate=0.001):
        self.classifier = MultiClassClassifier(input_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)

    def train(self, input_data, labels, num_epochs=10):
        losses = []
        import matplotlib.pyplot as plt

        plt.figure()
        for epoch in range(num_epochs):
            # Shuffle the input data and labels
            indices = torch.randperm(input_data.size(0))
            input_data = input_data[indices]
            labels = labels[indices]

            self.optimizer.zero_grad()
            outputs = self.classifier(input_data)
            loss = self.criterion(outputs, labels)

            if epoch % 100 == 0:
                losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        plt.plot(losses)
        plt.savefig('LOSSES.png')
    
    def predict(self, test_data):
        with torch.no_grad():
            predictions = self.classifier(test_data)
            predicted_labels = torch.argmax(predictions, dim=1)
        return predictions, predicted_labels




class ClfRunner:
    def __init__(self, hidden_dim, clf_key):
        self.hidden_dim = hidden_dim
        self.clf_key = clf_key
        self.results_dict = {}
        self.one_hot = OneHotEncoder(sparse_output=False)
        self.clf_split = 0.8

    def setup(self, mode, adata):
        self.mode = mode
        self.n_classes = adata.obs[self.clf_key].nunique()

        # produce integer labels
        str_label_categories = np.unique(np.array(adata.obs[self.clf_key]))
        category_to_index = {
            category: index
            for index, category in enumerate(str_label_categories)
        }
        vectorized_mapping = np.vectorize(category_to_index.get)
        
        self.lbls = vectorized_mapping(adata.obs[self.clf_key])

        self.graph_clf = NodeClassifier(self.hidden_dim, self.n_classes)

    def train_test_predictions(self, feats, labels, mode):
        """Train/test classifier"""

        if mode == 'Train':
            self.clf = ClassifierTrainer(feats.shape[-1], self.n_classes)   
            self.clf.train(
                feats,
                labels,
                num_epochs=10000
            )

        out, pred = self.clf.predict(feats)
        loss = self.clf.criterion(out, labels)

        labels_oh = self.one_hot.fit_transform(labels.reshape(-1, 1))
        pred_oh = self.one_hot.fit_transform(pred.reshape(-1, 1))

        acc = (torch.sum(pred==labels)/pred.shape[0]).item()
        
        print('Accuracy: ', acc)

        confusion_mat = sklearn.metrics.confusion_matrix(labels.numpy(), pred.numpy())

        self.results_dict.update({f"{mode} cross entropy": loss.item()})
        self.results_dict.update({f"{mode} classification accuracy": acc})
        #self.results_dict.update({f"{mode} classification precision": prec_acc})
        #self.results_dict.update({f"{mode} classification F1 score": f1_score})

        return confusion_mat

    def run_clf(self, adata, eval_key, emb_key='int'):
        """Run classifier"""
        all_idx = np.arange(adata.shape[0])
        np.random.shuffle(all_idx)
        train_idx, test_idx = (
            all_idx[: int(self.clf_split * len(all_idx))],
            all_idx[int(self.clf_split * len(all_idx)) :],
        )        

        # extract sample-specific labels from global label set
        labels = torch.tensor(self.lbls)
        try:
            feat = torch.tensor(adata.obsm[f"X_{emb_key}_{eval_key}"]).to(torch.float32)
        except:
            print('rep not found!')

        train_confusion = self.train_test_predictions(feat[train_idx][~torch.isnan(feat[train_idx].max(1).values)], labels[train_idx][~torch.isnan(feat[train_idx].max(1).values)], mode='Train')
        test_confusion = self.train_test_predictions(feat[test_idx][~torch.isnan(feat[test_idx].max(1).values)], labels[test_idx][~torch.isnan(feat[test_idx].max(1).values)], mode='Test')

        return [train_confusion, test_confusion]

    
    def plot_precision_recall_curve(self, y_score, outdir):
        from sklearn.metrics import average_precision_score, precision_recall_curve

        Y_test = self.one_hot.fit_transform(self.lbls["cell_type"].reshape(-1, 1))
        n_classes = Y_test.shape[-1]
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                Y_test[:, i], y_score[:, i]
            )
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            Y_test.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(
            Y_test, y_score, average="micro"
        )

        # setup plot details
        # n = 10  # Replace with your desired size 'n'
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

        # colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Multi-class Precision-Recall curve")

        plt.savefig(f"{outdir}/precision_recall.png")

    def fit_graph_clf(self, graph, x, lbl):
        """ "Fit node classifier"""
        for clf_key in self.clf_keys:
            train_mask = np.random.choice(
                graph.number_of_nodes(),
                int(graph.number_of_nodes() * 0.8),
                replace=False,
            )
            graph.ndata["train_mask"] = torch.zeros(len(graph.nodes()))
            graph.ndata["train_mask"][train_mask] = 1
            graph.ndata["val_mask"] = torch.zeros(len(graph.nodes()))
            graph.ndata["val_mask"][~train_mask] = 1
            optimizer = torch.optim.Adam(self.graph_clf[clf_key].parameters(), lr=0.01)
            best_val_acc = 0
            patience, current_patience = 10, 0

            for e in range(200):
                logits = self.graph_clf[clf_key](graph, x)
                pred = logits.argmax(1)

                loss = self.torch_loss_func(logits.to(torch.float32), lbl)
                train_acc = (
                    (pred[train_mask] == torch.tensor(lbl.argmax(1)[train_mask]))
                    .float()
                    .mean()
                )
                val_acc = (
                    (pred[~train_mask] == torch.tensor(lbl.argmax(1)[~train_mask]))
                    .float()
                    .mean()
                )

                if best_val_acc < val_acc:
                    current_patience = 0
                else:
                    current_patience += 1
                    if current_patience >= patience:
                        print(
                            "EARLY STOPPING AT EPOCH =%d VAL LOSS: %.4f"
                            % (epoch, val_loss)
                        )
                        break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # TODO: early stopping criterion?


# NOTE: can use a graph-based classifier later, if helpful for analysis
class NodeClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, n_classes):
        super().__init__()
        self.conv1 = GraphConv(hidden_channels, int(hidden_channels / 2))
        self.conv2 = GraphConv(int(hidden_channels / 2), n_classes)

    def forward(self, graph, x):
        x = self.conv1(graph, x)
        x = F.relu(x)
        x = self.conv2(graph, x)
        x = F.relu(x)
        return x
