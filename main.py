import argparse
import os
import os.path as osp
import pickle
import joblib
from urllib import request
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import data_loading as dl
import Models
from torchmetrics.functional import f1, accuracy, precision, recall, confusion_matrix
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
import time
import shutil
from data_loading import standardization
import matplotlib.pyplot as plt


CLASSES = {
    2: 'AwayWin',
    1: 'Draw',
    0: 'HomeWin',
}

"""def accuracy(prediction, labels_batch, dim=-1):
    pred_index = prediction.argmax(dim)
    return (pred_index == labels_batch).float().mean()"""


def learn(model, opt, trn_loader, tst_loader, max_epoch, verbose, device, param_reset, match_discount, epoch_no_improve,
          min_loss_delta, loss_queue_size, PATH):
    metric = MetricCollection([Accuracy(num_classes=3, average='macro'),
                               F1(num_classes=3, average='macro'),
                               Recall(num_classes=3, average='macro'),
                               Precision(num_classes=3, average='macro'),
                               ConfusionMatrix(num_classes=3)])
    # print(model.state_dict())
    # trn_epoch_stats = pd.DataFrame()#columns=pd.MultiIndex.from_product([["Epoch", "Batch"], ["Accuracy", "F1", "Recall", "Precision", "ConfusionMatrix", "Loss", "Epoch"]]))
    trn_batch_stats = pd.DataFrame()
    # print(trn_epoch_stats)
    # val_epoch_stats = pd.DataFrame()
    val_batch_stats = pd.DataFrame()  # columns=pd.MultiIndex.from_product([["Epoch", "Batch"], ["Accuracy", "F1", "Recall", "Precision", "ConfusionMatrix", "Loss", "Epoch"]]))
    if isinstance(model, Models.Random_forest):
        rf = True
    else:
        rf = False
    j = 0

    for k, (trn_batch, val_batch) in enumerate(trn_loader):
        start_all = time.perf_counter()
        start_cpu = time.process_time()
        """
        for m,match in enumerate(trn_batch['data']['skaters']):
            for t,team in enumerate(match):
                for match_val in val_batch['data']['skaters']:
                    for team_val in match_val:
                        #print(team.size(),team_val.size())
                        if torch.equal(team,team_val):
                            print(m,t)
        """

        if param_reset and not rf:
            model.apply(model_weight_reset)
            opt = torch.optim.Adam(model.parameters())
            print("param rest")
        print("Window position: %d/%d" % (j + 1, len(trn_loader)))

        if model.stat_type == "team":
            if trn_batch['data'].size()[1] != 2:
                trn_batch['data'] = standardization(trn_batch['data'], trn_batch['data'].size()[-1])
                val_batch['data'] = standardization(val_batch['data'], val_batch['data'].size()[-1])
        else:
            if 'none_zero' not in list(trn_batch.keys()):
                trn_batch["data"]["skaters"] = torch.cat((trn_batch["data"]["skaters"],val_batch["data"]["skaters"]),dim=0) #pro finalni testovani
                trn_batch["data"]["goalies"] = torch.cat((trn_batch["data"]["goalies"],val_batch["data"]["goalies"]),dim=0) #pro finalni testovani
                trn_skaters = standardization(trn_batch['data']['skaters'], trn_batch['data']['skaters'].size()[-1])
                trn_goalies = standardization(trn_batch['data']['goalies'], trn_batch['data']['goalies'].size()[-1])
                trn_batch['data'] = {
                    'skaters': trn_skaters,
                    'goalies': trn_goalies}
                val_skaters = standardization(val_batch['data']['skaters'], val_batch['data']['skaters'].size()[-1])
                val_goalies = standardization(val_batch['data']['goalies'], val_batch['data']['goalies'].size()[-1])
                val_batch['data'] = {
                    'skaters': val_skaters,
                    'goalies': val_goalies}
        trn_batch["labels"] = torch.cat((trn_batch["labels"],val_batch["labels"]),dim=0)
        val_batch = tst_loader.__next__()
        val_skaters = standardization(val_batch['data']['skaters'], val_batch['data']['skaters'].size()[-1])
        val_goalies = standardization(val_batch['data']['goalies'], val_batch['data']['goalies'].size()[-1])
        val_batch['data'] = {
            'skaters': val_skaters,
            'goalies': val_goalies}
        """
        trn_running_loss = 0
        trn_epoch_pred = torch.empty(0)
        trn_epoch_label = torch.empty(0).int()
        val_running_loss = 0
        val_epoch_pred = torch.empty(0)
        val_epoch_label = torch.empty(0).int()
        """
        i = 0
        min_avg_val_loss = np.inf
        loss_queue = []
        no_improve = 0
        early_stop = False
        sample_weight = torch.Tensor([1. + match_discount * i for i in range(len(trn_batch['labels']))])
        
        while not early_stop:
            i += 1

            print("%04d/%04d" % (i, max_epoch))  # , end="\r")
            # print(type(trn_batch),type(val_batch))
            # print(type(trn_batch),type(trn_batch["data"][0]),type(trn_batch["labels"][0]))
            # print(trn_batch["data"].shape,trn_batch["labels"].shape)
            if rf:
                early_stop = True
                if model.stat_type == "player":
                    data = trn_batch['data']
                else:
                    data = trn_batch['data'].numpy().copy()
                # print(data.shape)
                trn_label = trn_batch['labels'].numpy().copy()
                # print(trn_label.shape)
                model.fit(data, trn_label)
                trn_pred = torch.from_numpy(model.predict(data)).type(torch.FloatTensor)
                # print(trn_pred.type())

            if model.stat_type == "team":
                data = trn_batch['data'].to(device)
            else:
                try:
                    data = {"data": trn_batch['data'],
                            "none_zero": trn_batch['none_zero']}
                except:
                    data = trn_batch['data']

            trn_label = trn_batch['labels'].to(device)

            if not rf:
                model.train()
                trn_pred = model(data)
                # print(trn_pred.type())
            # print(trn_pred.argmax(dim=-1),trn_label)

            # print(sample_weight.size())
            loss = F.cross_entropy(trn_pred, trn_label, reduction="none")
            loss = (loss * sample_weight / sample_weight.sum()).sum().mean()



            if not rf:
                loss.backward()

            val_batch_metric, val_pred, val_label, val_loss = evaluate(model, len(CLASSES), val_batch, i, j,
                                                                       match_discount, device, verbose, PATH,
                                                                       save=False)

            # loss = F.cross_entropy(trn_pred, trn_label, reduction="mean")
            trn_batch_metric = metric(trn_pred.argmax(dim=-1), trn_label)
            # print(trn_batch_metric["Accuracy"])
            trn_batch_metric["Loss"] = F.cross_entropy(trn_pred, trn_label, reduction="mean").item()
            trn_batch_metric["Loss_weighted"] = loss.item()
            trn_batch_metric["Epoch"] = i
            trn_batch_metric["Window_position"] = j
            trn_batch_stats = trn_batch_stats.append(trn_batch_metric, ignore_index=True)

            """
            trn_running_loss += loss.item()
            trn_epoch_pred = torch.cat((trn_epoch_pred, trn_pred), dim=0)
            trn_epoch_label = torch.cat((trn_epoch_label, trn_label), dim=0)
            """

            if not rf:
                opt.step()
                opt.zero_grad()

            # print(val_batch_metric["Accuracy"])
            # print(pd.DataFrame.from_dict({"Batch": val_metric},orient="columns").unstack().to_frame().T["Batch"].set_index("Epoch"))
            if True:
                loss_queue.append(val_loss)
                if len(loss_queue) > loss_queue_size:
                    loss_queue.pop(0)
                average_loss = sum(loss_queue) / len(loss_queue)
                if min_avg_val_loss - average_loss <= min_loss_delta and i >= loss_queue_size:
                    no_improve += 1
                elif min_avg_val_loss - average_loss > min_loss_delta:
                    no_improve = 0
                min_avg_val_loss = average_loss
                # print(test_end, i, no_improve)
                if no_improve > epoch_no_improve or i == max_epoch:
                    early_stop = True
                # val_batch_metric["early_stop"] = early_stop
            # print(val_batch_stats)
            val_batch_stats = val_batch_stats.append(val_batch_metric, ignore_index=True)
            """
            val_running_loss += val_loss.item()
            val_epoch_pred = torch.cat((val_epoch_pred, val_pred), dim=0)
            val_epoch_label = torch.cat((val_epoch_label, val_label), dim=0)
            """
            if not rf:
                model.train()
        j += 1
        if rf:
            model_file_name = os.path.join(PATH, "last_model_for_window_pos_%d.joblib" % (k))
            joblib.dump(model.model, model_file_name)
        else:
            model_file_name = os.path.join(PATH, "last_model_for_window_pos_%d.pth" % (k))
            torch.save(model.state_dict(), model_file_name)
        # print(trn_epoch_pred.shape, trn_epoch_label.shape)
        # trn_epoch_pred = trn_epoch_pred.argmax(dim=-1)
        # print(trn_epoch_pred)
        # trn_epoch_metric = metric(trn_epoch_pred, trn_epoch_label)
        # trn_epoch_metric["Loss"] = trn_running_loss / j
        # trn_epoch_metric["Epoch"] = i + 1
        # trn_epoch_stats = trn_epoch_stats.append(trn_epoch_metric, ignore_index=True)

        # val_epoch_pred = val_epoch_pred.argmax(dim=-1)
        # val_epoch_metric = metric(val_epoch_pred, val_epoch_label)
        # val_epoch_metric["Loss"] = val_running_loss / j
        # val_epoch_metric["Epoch"] = i + 1
        # val_epoch_stats = val_epoch_stats.append(val_epoch_metric, ignore_index=True)
        # print(epoch_metric)
        # print('Train: Epoch: {:2d} / {:2d}\tAccuracy: {:.5f}\tLoss: {:.5f}'.format(i + 1, epochs, acc / len(trn_loader.dataset), running_loss/j))
        # f.write(str(i+1)+" | "+str(float(acc / len(trn_loader.dataset)))+" | "+str(float(running_loss/j))+" | ")
        print("CPU window", j, "time: ", time.process_time() - start_cpu)
        print("ALL window", j, "time: ", time.perf_counter() - start_all)

    # print(trn_stats["ConfusionMatrix"])
    # trn_epoch_stats.to_pickle(os.path.join(PATH, "train_epoch_stats.pkl"))
    # val_epoch_stats.to_pickle(os.path.join(PATH, "validate_epoch_stats.pkl"))
    print("trn")
    print(trn_batch_stats['ConfusionMatrix'])
    print("val")
    print(val_batch_stats['ConfusionMatrix'])
    trn_batch_stats.to_pickle(os.path.join(PATH, "train_all_stats.pkl"))
    val_batch_stats.to_pickle(os.path.join(PATH, "validate_all_stats.pkl"))
    df_vl = val_batch_stats["Loss"][:1000].astype(float)
    df_vl.plot(color='b')
    df_vlw = val_batch_stats["Loss_weighted"][:1000].astype(float)
    df_vlw.plot(color='m', lw=0.5)
    df_tl = trn_batch_stats["Loss"][:1000].astype(float)
    df_tl.plot(color='y')
    df_tlw = trn_batch_stats["Loss_weighted"][:1000].astype(float)
    fig = df_tlw.plot(color='r', lw=0.5)

    plt.show()

'''
def evaluate(model, val_loader, device):
    num_classes = 10
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=device
    )  # empty confusion matrix
    orders = []
    with torch.no_grad():
        acc = 0
        for i, batch in enumerate(val_loader):
            data = batch['data'].to(device)
            labels = batch['labels'].to(device)

            prediction = model(data)
            confusion_matrix += create_confusion_matrix(num_classes, prediction, labels)
            order = get_prediction_order(prediction, labels).cpu().numpy()
            orders.append(order)
            acc += accuracy(prediction, labels) * data.shape[0]

        acc1='{:.3f}'.format(acc/len(val_loader.dataset)*100)
        global best_acc
        print('Evaluate: accuracy: {:.5f} best accuracy: {:.5f}'.format(acc / len(val_loader.dataset),best_acc))

        if float(acc1) > best_acc and float(acc1) > 35:
            best_acc = float(acc1)
            PATH = 'modely/'+ str(acc1) + '%.pt'
            torch.save(model.state_dict(), PATH)

    orders = np.concatenate(orders, 0)
    confusion_matrix = confusion_matrix.cpu().numpy()
    #print_stats(confusion_matrix, orders)
    print(confusion_matrix)
'''


def evaluate(model, num_classes, loader, epoch, window_position, match_discount, device, verbose, PATH, save):
    if isinstance(model, Models.Random_forest):
        rf = True
    else:
        rf = False
    if not rf:
        model = model.to(device)
        model = model.eval()  # switch to eval mode, so that some special layers behave nicely
    # confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)  # empty confusion matrix
    # orders = []
    # train_metrics = torchmetrics.MetricCollection
    metric = MetricCollection([Accuracy(num_classes=num_classes, average='macro'),
                               F1(num_classes=num_classes, average='macro'),
                               Recall(num_classes=num_classes, average='macro'),
                               Precision(num_classes=num_classes, average='macro'),
                               ConfusionMatrix(num_classes=num_classes)])
    # print(loader)
    with torch.no_grad():  # disable gradient computation
        epoch_pred = torch.empty(0)
        epoch_label = torch.empty(0).int()
        if rf:
            if model.stat_type == "player":
                data = loader['data']
            else:
                data = loader['data'].numpy().copy()
            prediction = torch.from_numpy(model.predict(data))

        if model.stat_type == "team":
            data = loader['data'].to(device)
        else:
            try:
                data = {"data": loader['data'],
                        "none_zero": loader['none_zero']}
            except:
                data = loader['data']
        labels = loader['labels'].to(device)
        if not rf:
            prediction = model(data)
        sample_weight = torch.Tensor([1. + match_discount * i for i in range(len(loader['labels']))])
        loss = F.cross_entropy(prediction, labels, reduction="none")
        loss = (loss * sample_weight / sample_weight.sum()).sum().mean()
        epoch_pred = torch.cat((epoch_pred, prediction), dim=0)
        epoch_label = torch.cat((epoch_label, labels), dim=0)
        epoch_metric = metric(epoch_pred.argmax(dim=-1), epoch_label)
        epoch_metric["Loss"] = F.cross_entropy(prediction, labels, reduction="mean")
        epoch_metric["Loss_weighted"] = loss
        epoch_metric["Epoch"] = epoch
        epoch_metric["Window_position"] = window_position
        # print(epoch_metric["Epoch"])

        # print('Evaluate: accuracy: {:.3f} best accuracy: {:.3f}\t epoch: {:d} / {:d}'.format(acc / len(loader.dataset), best_acc, epoch, epochs))
    """if save:
        tst_stats = pd.DataFrame()
        tst_stats.append(epoch_metric, ignore_index=True)
        tst_stats.to_pickle(os.path.join(PATH, "test_stats.pkl"))"""
    return epoch_metric, epoch_pred, epoch_label, loss


def create_confusion_matrix(num_classes, prediction, label):
    prediction = prediction.detach()
    label = label.detach()
    prediction = torch.argmax(prediction, 1)
    cm = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=label.device
    )  # empty confusion matrix
    indices = torch.stack((label, prediction))  # stack labels and predictions
    new_indices, counts = torch.unique(indices, return_counts=True,
                                       dim=1)  # Find, how many cases are for each combination of (pred, label)
    cm[new_indices[0], new_indices[1]] += counts

    return cm


def get_prediction_order(prediction, label):
    prediction = prediction.detach()  # detach from computational graph (no grad)
    label = label.detach()

    prediction_sorted = torch.argsort(prediction, 1, True)
    finder = (label[:, None] == prediction_sorted)
    order = torch.nonzero(finder)[:, 1]  # returns a tensor of indices, where finder is True.

    return order


def print_stats(conf_matrix, orders, file_name):
    num_classes = conf_matrix.shape[0]
    with open(file_name, "a") as file:
        file.write("----------\n")
        file.write('Confusion matrix:\n')
        for i, r in enumerate(conf_matrix):
            for c in r:
                file.write(str(c) + " ")
            if i + 1 != len(conf_matrix[-1]):
                file.write("\n")
        file.write("\n")
        file.write('Precision and recalls:\n')
        for c in range(num_classes):
            precision = conf_matrix[c, c] / conf_matrix[:, c].sum()
            recall = conf_matrix[c, c] / conf_matrix[c].sum()
            f1 = (2 * precision * recall) / (precision + recall)
            file.write('Class {cls:10s} ({c}):\tPrecision: {prec:0.5f}\tRecall: {rec:0.5f}\tF1: {f1:0.5f}\n'.format(
                cls=CLASSES[c], c=c, prec=precision, rec=recall, f1=f1))
        file.write('Top-n accuracy and error:\n')
        order_len = len(orders)
        for n in range(num_classes):
            topn = (orders <= n).sum()
            acc = topn / order_len
            err = 1 - acc
            file.write('Top-{n}:\tAccuracy: {acc:0.5f}\tError: {err:0.5f}\n'.format(n=(n + 1), acc=acc, err=err))


def model_weight_reset(model):
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()


def model_weight_print(model):
    model_weight = getattr(model, "weight", None)
    if model_weight is not None:
        print(model_weight)


def model_weight_set(model):
    if hasattr(model, "weight"):
        if hasattr(model.weight, "fill_"):
            model.weight.data.fill_(2.)


def parse_args():
    parser = argparse.ArgumentParser('NHL match predictor')
    parser.add_argument('--stat_type', '-st', default="player", type=str)
    parser.add_argument('--granularity', '-g', default="High", type=str)
    parser.add_argument('--embedding', '-emb', default=0, type=int)
    parser.add_argument('--num_of_train_year', '-ntrn', default=16, type=int)
    parser.add_argument('--window_size', '-ws', default=15, type=int)
    parser.add_argument('--end_year', '-ey', default=2019, type=int)
    parser.add_argument('--model', '-mod', default="normal", type=str)
    parser.add_argument('--layers', '-l', default=4, type=int)
    parser.add_argument('--match_discounting', '-md', default=0.0, type=float)
    parser.add_argument('--max_epochs', '-me', default=10000, type=int)
    parser.add_argument('--epoch_no_improve', '-enp', default=10, type=int)
    parser.add_argument('--loss_queue_size', '-lqs', default=20, type=int)
    parser.add_argument('--loss_delta', '-ld', default=0.0, type=float)
    parser.add_argument('--param_reset', '-pr', default="True", type=str)
    parser.add_argument('--dim_reduction', '-dr', default="False", type=str)
    parser.add_argument('--batch_size', '-bs', default=1000, type=int)
    parser.add_argument('--optimiser', '-op', default="adam", type=str)
    parser.add_argument('--learning_rate', '-lr', default=0.008, type=float)
    parser.add_argument('--weight_decay', '-wd', default=0.41, type=float)
    parser.add_argument('--momentum', '-mom', default=0.05, type=float)
    parser.add_argument('--run_location', '-rl', default="server", type=str)
    parser.add_argument('--verbose', '-v', default=True, action='store_true')
    parser.add_argument('--database_location', '-dl', default="remote", type=str)
    parser.add_argument('--start_script_name', '-ssn', default="0000", type=str)
    # parser.add_argument('--store_dir', '-sd', default='data', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.param_reset == "False":
        args.param_reset = False
    else:
        args.param_reset = True
    if args.granularity != "Low":
        args.embedding = 0
    name = ""
    for key, value in vars(args).items():
        name += str(value) + "_"
    file_name = args.start_script_name + ".txt"
    name = name[:-6]
    open(os.path.join("scripts", "loading", file_name), 'a').close()
    PATH = "test"
    finished_computations = os.listdir(PATH)

    if name in finished_computations:
        PATH = os.path.join(PATH, name)
        if "validate_all_stats.pkl" in os.listdir(PATH):
            print("skipped")
            shutil.move(os.path.join("scripts", "loading", file_name), os.path.join("scripts", "skipped", file_name))
            open(os.path.join(PATH, file_name), 'a').close()
            exit()
    else:
        PATH = os.path.join(PATH, name)
        if not os.path.exists(PATH):
            os.mkdir(PATH)

    # open(os.path.join(PATH, file_name), 'a').close()
    if args.stat_type == "team":
        trn_loader, tst_loader, stat_len = dl.load_team_data(args.num_of_train_year, args.window_size, args.end_year,
                                                             args.granularity, args.database_location,
                                                             args.run_location, file_name)
    elif args.stat_type == "player":
        trn_loader, tst_loader, stat_len = dl.load_player_data(args.num_of_train_year, args.window_size, args.end_year,
                                                               args.granularity, args.database_location,
                                                               args.run_location, file_name)

    device = 'cpu'

    if args.model == "normal":
        model = Models.Model_team_N_layer(stat_len, len(CLASSES), args.layers, args.embedding, args.dim_reduction).to(device)
    elif args.model == "logistical":
        model = Models.Model_team_logistical(stat_len, len(CLASSES), args.embedding, args.dim_reduction).to(device)
    elif args.model == "random_forest":
        model = Models.Random_forest(args.stat_type, stat_len)

    if args.model != "random_forest":
        if args.optimiser == "adam":
            opt = torch.optim.Adam(model.parameters())  # , lr=args.learning_rate)
        elif args.optimiser == "sgd":
            opt = torch.optim.SGD(model.parameters(), momentum=args.momentum, weight_decay=args.weight_decay,
                                  lr=args.learning_rate)
    else:
        opt = None

    learn(model, opt, trn_loader, tst_loader, args.max_epochs, args.verbose, device, args.param_reset,
          args.match_discounting,
          args.epoch_no_improve, args.loss_delta, args.loss_queue_size, PATH)
    shutil.move(os.path.join("scripts", "loaded", file_name), os.path.join("scripts", "finished", file_name))


    # DON'T LOOK AT THIS BEFORE EXPERIMENTS
    """ 
    best_loss = np.inf
    tst_stats = pd.DataFrame()
    for name in os.listdir(PATH):
        if "best_loss_model" in name:
            if args.model == "random_forest":
                model.model = joblib.load(os.path.join(PATH,name))
                for i, loader in enumerate(tst_loader):
                    tst_metric = evaluate(model, len(CLASSES), loader, None, None, None, device, False, PATH, save=True)
                if tst_metric[0]["Loss"] < best_loss:
                    best_loss = tst_metric[0]["Loss"]
                    model_file_name = os.path.join(PATH, "best_model_on_tst.joblib")
                    joblib.dump(model.model, model_file_name)
            else:
                model.load_state_dict(torch.load(os.path.join(PATH, name)))
                for i, loader in enumerate(tst_loader):
                    tst_metric = evaluate(model, len(CLASSES), loader, None, None, None, device, False, PATH, save=True)
                if tst_metric[0]["Loss"] < best_loss:
                    best_loss = tst_metric[0]["Loss"]
                    model_file_name = os.path.join(PATH, "best_model_on_tst.pth")
                    torch.save(model.state_dict(), model_file_name)
            tst_stats = tst_stats.append(tst_metric[0], ignore_index=True)
    tst_stats.to_pickle(os.path.join(PATH, "test_stats.pkl"))
    """


best_loss = 0
if __name__ == '__main__':
    main()
