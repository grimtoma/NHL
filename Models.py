import torch.nn
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_loading import standardization
#from sklearn.preprocessing import StandardScaler


class Random_forest():
    def __init__(self, stat_type, stat_len):
        self.model = RandomForestClassifier()
        self.stat_type = stat_type
        self.stat_len = stat_len

    def fit(self, data, labels):
        if self.stat_type == "team":
            self.model.fit(data, labels)
        else:
            skaters = data['skaters'].numpy()
            goalies = data['goalies'].numpy()
            skaters = np.mean(skaters, axis=2)
            goalies = np.mean(goalies, axis=2)
            players = np.concatenate((skaters, goalies), axis=2)
            num_of_matches = players.shape[0]
            x = np.reshape(players, (num_of_matches, -1))
            self.model.fit(x, labels)

    def predict(self, data):
        if self.stat_type == "team":
            return self.model.predict_proba(data)
        else:
            skaters = data['skaters'].numpy()
            goalies = data['goalies'].numpy()
            skaters = np.mean(skaters, axis=2)
            goalies = np.mean(goalies, axis=2)
            players = np.concatenate((skaters, goalies), axis=2)
            num_of_matches = players.shape[0]
            x = np.reshape(players, (num_of_matches, -1))
            return self.model.predict_proba(x)


class Model_team_logistical(torch.nn.Module):
    def __init__(self, stat_len, num_of_classes, embedding, dim_reduction):
        super().__init__()
        self.dim_reduction = dim_reduction
        if type(stat_len) is tuple:
            self.skater_reduced_dimension = stat_len[0] // 3
            self.goalie_reduced_dimension = stat_len[1] // 3
            self.player_stat_len = stat_len
            self.stat_type = "player"
            self.conv_skater = torch.nn.Conv2d(1, self.skater_reduced_dimension, kernel_size=(1, stat_len[0]))
            self.conv_goalie = torch.nn.Conv2d(1, self.goalie_reduced_dimension, kernel_size=(1, stat_len[1]))
            if self.dim_reduction:
                self.stat_len = 2 * (self.skater_reduced_dimension + self.goalie_reduced_dimension)
            else:
                self.stat_len = 2 * (stat_len[0] + stat_len[1])
        else:
            self.stat_type = "team"
            self.stat_len = stat_len
        self.num_of_classes = num_of_classes
        if embedding:
            if self.stat_type == "team":
                self.embedding = torch.nn.Embedding(stat_len, embedding)
            else:
                self.embedding = torch.nn.Embedding(stat_len[0], embedding, padding_idx=0)
            self.stat_len = 2 * embedding
        else:
            self.embedding = False
        self.tanh = torch.nn.Tanh()
        self.batch = torch.nn.BatchNorm1d(self.stat_len,track_running_stats=False)
        self.lin = torch.nn.Linear(self.stat_len, self.num_of_classes)

    def forward(self, x):
        if self.stat_type == "player":
            if not self.embedding:
                if self.dim_reduction:
                    skaters = x["skaters"]
                    goalies = x["goalies"]
                    skaters = skaters.view(-1, 1, skaters.size()[2], self.player_stat_len[0])
                    goalies = goalies.view(-1, 1, goalies.size()[2], self.player_stat_len[1])
                    skaters = self.conv_skater(skaters)
                    goalies = self.conv_goalie(goalies)
                    skaters = torch.mean(skaters, 2)
                    goalies = torch.mean(goalies, 2)
                    x = torch.cat((skaters, goalies), dim=1)
                else:
                    skaters = x["skaters"]
                    goalies = x["goalies"]
                    skaters = torch.mean(skaters, 2)
                    goalies = torch.mean(goalies, 2)
                    x = torch.cat((skaters, goalies), dim=2)
            else:
                none_zero = x["none_zero"]
                x = x["data"]
                x = self.embedding(x)
                x = torch.sum(x, dim=2)
                x = x / none_zero
        else:
            if self.embedding:
                # print(x[:, 0].type(torch.LongTensor))
                # print(x.shape, x[:, 0].shape)
                x = torch.cat(
                    (self.embedding(x[:, 0].type(torch.LongTensor)), self.embedding(x[:, 1].type(torch.LongTensor))),
                    dim=1)
                # print(x.shape)
        x = x.view(-1, self.stat_len)
        x = self.lin(self.tanh(x))#self.batch(x)))
        return x


class Model_team_N_layer(torch.nn.Module):
    def __init__(self, stat_len, num_of_classes, number_of_layers, embedding, dim_reduction):
        super().__init__()
        self.dim_reduction = dim_reduction
        if type(stat_len) is tuple:
            self.skater_reduced_dimension = stat_len[0] // 3
            self.goalie_reduced_dimension = stat_len[1] // 3
            self.player_stat_len = stat_len
            self.stat_type = "player"
            self.conv_skater = torch.nn.Conv2d(1, self.skater_reduced_dimension, kernel_size=(1, stat_len[0]))
            self.conv_goalie = torch.nn.Conv2d(1, self.goalie_reduced_dimension, kernel_size=(1, stat_len[1]))
            self.lin_skater = torch.nn.Linear(stat_len[0], self.skater_reduced_dimension)
            self.lin_goalie = torch.nn.Linear(stat_len[1], self.goalie_reduced_dimension)
            self.pool_skater = torch.nn.AvgPool2d(kernel_size=(33, 1), stride=None, padding=0, ceil_mode=False)
            self.pool_goalie = torch.nn.AvgPool2d(kernel_size=(5, 1), stride=None, padding=0, ceil_mode=False)
            if self.dim_reduction:
                self.stat_len = 2 * (self.skater_reduced_dimension + self.goalie_reduced_dimension)
            else:
                self.stat_len = 2 * (stat_len[0] + stat_len[1])
        else:
            self.stat_type = "team"
            self.stat_len = stat_len
        self.num_of_classes = num_of_classes
        if embedding:
            if self.stat_type == "team":
                # print(stat_len, embedding)
                self.embedding = torch.nn.Embedding(stat_len, embedding)
            else:
                self.embedding = torch.nn.Embedding(stat_len[0], embedding, padding_idx=0)
            self.stat_len = 2 * embedding
        else:
            self.embedding = False
        self.num_of_layers = number_of_layers
        self.tanh = torch.nn.Tanh()
        self.batchs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        last_size = self.stat_len
        size = 0
        for i in range(1, self.num_of_layers):
            print("D:", self.stat_len, self.num_of_classes, self.num_of_layers, i)
            size = round(((self.stat_len - self.num_of_classes) / (self.num_of_layers)) * (
                        self.num_of_layers - i)) + self.num_of_classes
            # self.batchs.append(torch.nn.BatchNorm1d(last_size,track_running_stats=False))
            self.lins.append(torch.nn.Linear(last_size, size))
            print(last_size, size)
            last_size = size

        #self.batchs.append(torch.nn.BatchNorm1d(last_size))
        self.lins.append(torch.nn.Linear(last_size, self.num_of_classes))

    def forward(self, x):
        if self.stat_type == "player":
            if not self.embedding:
                if self.dim_reduction:
                    skaters = x["skaters"]
                    goalies = x["goalies"]
                    # print(skaters.size())
                    # print(goalies.size())
                    skaters = skaters.view(-1, 1, skaters.size()[2], self.player_stat_len[0])
                    goalies = goalies.view(-1, 1, goalies.size()[2], self.player_stat_len[1])
                    # print(skaters.size())
                    # print(goalies.size())
                    # skaters = self.conv_skater(skaters)
                    # goalies = self.conv_goalie(goalies)
                    skaters = self.lin_skater(skaters)
                    goalies = self.lin_goalie(goalies)
                    # print(self.lin_skater.weight.data)
                    # print(skaters.size())
                    # print(goalies.size())
                    # skaters = self.pool_skater(skaters)
                    # goalies = self.pool_goalie(goalies)
                    skaters = torch.mean(skaters, 2).squeeze()
                    goalies = torch.mean(goalies, 2).squeeze()
                    # print(skaters.size())
                    # print(goalies.size())
                    x = torch.cat((skaters, goalies), dim=1)

                else:
                    skaters = x["skaters"]
                    goalies = x["goalies"]
                    skaters = torch.mean(skaters, 2)
                    goalies = torch.mean(goalies, 2)
                    x = torch.cat((skaters, goalies), dim=2)
            else:
                none_zero = x["none_zero"]
                x = x["data"]
                x = self.embedding(x)
                x = torch.sum(x, dim=2)
                x = x / none_zero
        else:
            if self.embedding:
                # print(x[:, 0].type(torch.LongTensor))
                # print(x.shape, x[:, 0].shape)
                x = torch.cat(
                    (self.embedding(x[:, 0].type(torch.LongTensor)), self.embedding(x[:, 1].type(torch.LongTensor))),
                    dim=1)
                # print(x.shape)
        x = x.view(-1, self.stat_len)
        game = x[:, 0].tolist()
        for i in range(self.num_of_layers):
            x = self.lins[i](self.tanh(x))#standardization(x,x.size()[-1])))#self.batchs[i](x)))
        return x
