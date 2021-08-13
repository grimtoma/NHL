import numpy
import torch.utils.data
import numpy as np
import pandas as pd
from database_access import psycopg2_connection_to_database, psycopg2_connection_to_remote_database, \
    sqlalchemy_connection_to_database, sqlalchemy_connection_to_remote_database
import time
import shutil
import os
from sklearn.preprocessing import StandardScaler


def standardization(x, stat_len):
    """x = x.numpy()
    shape = np.shape(x)
    x = np.reshape(x, (-1, stat_len))
    scaler = StandardScaler()
    scaler.fit(x)
    x_stand = scaler.transform(x)
    x_stand = torch.from_numpy(np.reshape(x_stand,shape))"""
    m = x.view(-1, stat_len).mean(-2, keepdim=True)
    s = x.view(-1, stat_len).std(-2, unbiased=False, keepdim=True)
    x_stand = (x - m) / s
    safe_tensor = torch.where(torch.isnan(x_stand), torch.zeros_like(x_stand), x_stand)
    return safe_tensor


def get_column_names(table_name, location, run_location):
    if location == "local":
        con = psycopg2_connection_to_database()
    elif location == "remote":
        con, server = psycopg2_connection_to_remote_database(run_location)
    column_names = []
    with con.cursor() as cur:
        cur.execute("SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '%s'" % (table_name))
        columns = cur.fetchall()
        for tuple in columns:
            column_names += [tuple[0]]
        con.close()
    if location == "remote":
        server.stop()
    return column_names


def load_data(batch_size, num_of_train_year, window_size, end_year, granularity, database_location, run_location,
              epochs):
    engine, server = sqlalchemy_connection_to_remote_database()
    column_names = get_column_names()
    table_name = ("cumulative_team_stats", "game_stats")
    with engine.connect() as con:
        query = 'SELECT g."teamIdHome",g."teamIdAway",g."goalsHome",g."goalsAway",%s FROM %s AS t \
                     JOIN %s AS g ON t."gameId" = g."gameId" \
                     WHERE t."gameId" >= %s AND t."gameId" < %s \
                     ORDER BY t."timeDate" ASC, t."gameId" ASC' \
                % (",".join(f't."{name}"' for name in column_names), table_name[0], table_name[1],
                   int(str(start_year) + "020000"), int(str(end_year + 1) + "020000"))
        df = pd.read_sql(query, con)

    column_name = get_column_names('cumulative_player_stats_4')
    START_YEAR = 2000
    cur.execute(
        'SELECT %s FROM cumulative_player_stats_4 AS p JOIN game_stats AS g ON p.game = g."gameId" WHERE p.game >= %s ORDER BY g."timeDate"' % (
            ",".join(f'p.{name}' for name in column_name), int(str(START_YEAR) + "020000")))
    data = cur.fetchall()
    trn_data = []
    val_data = []
    highest_id = 0
    for match in data:
        if match[1] > highest_id:
            highest_id = match[1]
        if match[2] > highest_id:
            highest_id = match[2]
        if int(str(match[0])[:4]) < END_YEAR:
            trn_data.append(match)
        else:
            val_data.append(match)
    print("trn")
    trn_dataset = Dataset(trn_data)
    print("val")
    print("highest", highest_id)
    val_dataset = Dataset(val_data)
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    cur.close()
    con.close()
    # print("Connection to database is CLOSED.")
    return trn_loader, val_loader


def load_player_data(num_of_train_year, window_size, end_year, granularity, database_location, run_location, file_name):
    """
    if location == "local":
        con = psycopg2_connection_to_database()
        table_name = 'cumulative_player_stats_6'
    elif location == "remote":
        con, server = psycopg2_connection_to_remote_database(run_location)
        table_name = 'cumulative_player_stats'
    column_names = get_column_names(table_name, location,run_location)
    max_length = 0
    game_numbers = []
    start_train_year = end_year - num_of_test_year - num_of_val_year - num_of_train_year
    start_val_year = end_year - num_of_test_year - num_of_val_year
    start_test_year = end_year - num_of_test_year
    with con.cursor() as cur:
    """
    start_all = time.perf_counter()
    start_cpu = time.process_time()

    if database_location == "local":
        con = psycopg2_connection_to_database()
        table_name = ("cumulative_player_stats_6", "game_stats")
    elif database_location == "remote":
        con, server = psycopg2_connection_to_remote_database(run_location)
        table_name = ("cumulative_player_stats", "game_stats")
    column_names = get_column_names(table_name[0], database_location, run_location)
    game_numbers = {}
    max_length = [0, 0]
    # print(num_of_train_year)
    start_year = end_year - num_of_train_year
    if start_year <= 2004:
        start_year -= 1  # -1 protoze v roce 2004 byla vyluka a nehral se jediny zapas
    print(start_year)
    with con.cursor() as cur:
        cur.execute('SELECT pi."primaryPosition",%s FROM %s AS p \
                     JOIN %s AS g ON p.game = g."gameId" \
                     JOIN player_info AS pi ON p.player = pi.id \
                     WHERE p.game >= %s AND p.game < %s \
                     ORDER BY g."timeDate" ASC , p.game ASC, p.side ASC, p.player ASC' # ORDER BY g."timeDate" ASC , p.game ASC, p.side ASC, p.player ASC'
                    % (",".join(f'p."{name}"' for name in column_names), table_name[0], table_name[1],
                       int(str(start_year) + "020000"),
                       int(str(end_year + 1) + "020000")))
        data = cur.fetchall()
    con.close()
    if database_location == "remote":
        server.stop()
    # print(len(data),len(data[0]))
    grouped_data = []
    labels = []
    game_idx = column_names.index('game') + 1
    team_idx = column_names.index('team') + 1
    player_idx = column_names.index('player') + 1
    result_idx = column_names.index('results') + 1
    side_idx = column_names.index('side') + 1
    goals_per_minute_idx = column_names.index('goalsPerMinute') + 1
    goalie_stats = ['timeOnIce', 'pim', 'shortHandedShotsAgainst', 'powerPlayShotsAgainst', 'evenShotsAgainst', 'saves',
                    'shots', 'assists', 'goals', 'shortHandedSaves', 'powerPlaySaves', 'evenSaves', 'goalsPerMinute',
                    'shotsPerMinute', 'assistsPerMinute', 'pimPerMinute', 'goalsPerGame', 'shotsPerGame',
                    'assistsPerGame', 'pimPerGame', 'timeOnIcePerGame', 'savesPerMinute', 'savesPerGame',
                    'savesPercentage', 'powerPlaySavesPerGame', 'powerPlaySavesPercentage', 'shortHandedSavesPerGame',
                    'shortHandedSavesPercentage', 'evenSavesPerGame', 'evenSavesPercentage']
    skater_stats = ['timeOnIce', 'takeaways', 'faceoffTaken', 'shortHandedAssists', 'faceOffWins', 'powerPlayTimeOnIce',
                    'shots', 'assists', 'goals', 'plusMinus', 'hits', 'shortHandedGoals', 'evenTimeOnIce', 'blocked',
                    'goalsPerMinute', 'shotsPerMinute', 'assistsPerMinute', 'powerPlayGoals', 'powerPlayAssists',
                    'goalsPerGame', 'shotsPerGame', 'assistsPerGame', 'timeOnIcePerGame', 'shortHandedTimeOnIce',
                    'giveaways', 'hitsPerMinute', 'goalsPercentage', 'takeawaysPerMinute', 'giveawaysPerMinute',
                    'blockedPerMinute', 'powerPlayGoalsPerMinute', 'powerPlayAssistsPerMinute',
                    'shortHandedGoalsPerMinute', 'shortHandedAssistsPerMinute', 'evenGoalsPerMinute',
                    'evenAssistsPerMinute', 'hitsPerGame', 'takeawaysPerGame', 'giveawaysPerGame', 'blockedPerGame',
                    'powerPlayGoalsPerGame', 'powerPlayAssistsPerGame', 'shortHandedGoalsPerGame',
                    'shortHandedAssistsPerGame', 'evenGoalsPerGame', 'evenAssistsPerGame', 'faceoffPercentage',
                    'powerPlayTimeOnIcePerGame', 'shortHandedTimeOnIcePerGame', 'teamGoalsPercentage',
                    'teamShotsPercentage']  # , 'teamBlockedPercentage', 'teamTakeawaysPercentage', 'teamHitsPercentage']
    if granularity == "Medium":
        temp = []
        for stat in goalie_stats:
            if "Per" in stat:
                continue
            else:
                temp.append(stat)
        goalie_stats = temp
        temp = []
        for stat in skater_stats:
            if "Per" in stat:
                continue
            else:
                temp.append(stat)
        skater_stats = temp
    temp = []
    for stat in skater_stats:
        if "hits" in stat:
            continue
        elif "takeaways" in stat:
            continue
        elif "giveaways" in stat:
            continue
        elif "blocked" in stat:
            continue
        else:
            temp.append(stat)
    skater_stats = temp
    print(skater_stats)
    col_names = get_column_names(table_name[0], database_location, run_location)
    goalie_stats_index = np.where(np.isin(col_names, goalie_stats))[0] + 1
    skater_stats_index = np.where(np.isin(col_names, skater_stats))[0] + 1
    player_stats_index = (skater_stats_index, goalie_stats_index)
    # print(goalie_stats_index)
    last_side = None
    if __name__ != '__main__':
        shutil.move(os.path.join("scripts", "loading", file_name), os.path.join("scripts", "loaded", file_name))

    """
    data1 = data
    start = time.process_time()
    
    data = np.array(data)
    keys = np.unique(data[:, 4])
    reindex_player_id = dict(zip(keys, [i for i in range(len(keys))]))
    """
    print("CPU loading: ", time.process_time() - start_cpu)
    print("ALL loading: ", time.perf_counter() - start_all)
    start_all = time.perf_counter()
    start_cpu = time.process_time()

    reindex_player_id = {}
    i = 1
    for item in data:
        player_id = item[player_idx]
        if player_id not in reindex_player_id:
            reindex_player_id[player_id] = i
            i += 1
    # print("CPU reindex: ", time.process_time() - start_cpu)
    # print("ALL reindex: ", time.perf_counter() - start_all)

    # print(reindex_player_id)
    average_player = []
    i = -1
    for item in data:
        # print(item[decision_stat_idx])
        if item[0] == "Goalie":
            goalie_flag = 1
        else:
            goalie_flag = 0

        if item[game_idx] not in game_numbers:
            i += 1
            game_numbers[item[game_idx]] = i
            #print(game_numbers)
            # game_numbers.append(item[game_idx])
            # game_numbers.append(i)

            if item[side_idx] == 0:
                labels.append(item[result_idx])
            if granularity == "Low":
                grouped_data.append([[],
                                     []])
            else:
                grouped_data.append([[[], []],  #
                                     [[], []]])  # +6s
                average_player.append([[[0] * len(player_stats_index[0]), [0] * len(player_stats_index[1])],  #
                                       [[0] * len(player_stats_index[0]), [0] * len(player_stats_index[1])]])  #

        if granularity == "Low":
            grouped_data[-1][item[side_idx]].append(reindex_player_id[item[player_idx]])
            # print(item[player_idx])
        elif granularity == "Medium":
            # print(list(item[player_stats_index[goalie_flag]].astype(np.float32)))
            stats = [item[index] for index in player_stats_index[goalie_flag]]  # +2s

            grouped_data[-1][item[side_idx]][goalie_flag].append(stats)  # +1s
            # print(type(item[player_stats_index[goalie_flag]].astype(np.float32).tolist()))
            # print(item[game_idx],item[side_idx])
            # print(average_player[-1][int(item[side_idx])][goalie_flag])
            # print(list(item[player_stats_index[goalie_flag]].astype(np.float32)))
            # print(stats)
            # print(item[game_idx],list(game_numbers.keys())[list(game_numbers.values()).index(i)])
            # print(average_player[-1][int(item[side_idx])][goalie_flag])
            # print(stats)
            average_player[-1][item[side_idx]][goalie_flag] = [a + b for a, b in
                                                               zip(average_player[-1][item[side_idx]][goalie_flag],
                                                                   stats)]
            # print(average_player[-1][int(item[side_idx])][goalie_flag])
            # print(average_player[-1][int(item[side_idx])][goalie_flag])
        elif granularity == "High":
            stats = [item[index] for index in player_stats_index[goalie_flag]]  # +2s
            #stats[0] = int(str(item[game_idx])[2:4]+str(item[game_idx])[-4:])
            grouped_data[-1][item[side_idx]][goalie_flag].append(stats)  # +1s
            average_player[-1][item[side_idx]][goalie_flag] = [a + b for a, b in
                                                               zip(average_player[-1][item[side_idx]][goalie_flag],
                                                                   stats)]

            # grouped_data[-1][int(item[side_idx])][goalie_flag].append(item[player_stats_index[goalie_flag]].astype(np.float32).tolist())
            # average_player[-1][int(item[side_idx])][goalie_flag] = [a + b for a, b in zip(average_player[-1][int(item[side_idx])][goalie_flag],item[player_stats_index[goalie_flag]].astype(np.float32).tolist())]

        if granularity == "Low":
            if max_length[0] < len(grouped_data[-1][item[side_idx]]):
                max_length[0] = len(grouped_data[-1][item[side_idx]])
        else:
            if max_length[goalie_flag] < len(grouped_data[-1][item[side_idx]][goalie_flag]):
                max_length[goalie_flag] = len(grouped_data[-1][item[side_idx]][goalie_flag])

    print("CPU grouped: ", time.process_time() - start_cpu)
    print("ALL grouped: ", time.perf_counter() - start_all)
    start_all = time.perf_counter()
    start_cpu = time.process_time()

    # print(grouped_data[-1])
    # print(len(grouped_data),len(grouped_data[-1]),len(grouped_data[-1][item[side_idx]]))

    # print(type(grouped_data[0]), type(grouped_data[0][0]), type(grouped_data[0][0][0]), type(grouped_data[0][0][0][0]), type(grouped_data[0][0][0][0][0]))
    # print(self.max_length)
    # stat_len = len(grouped_data[0][0][0])
    # print(max_length)
    none_zero = []
    for g, game in enumerate(grouped_data):
        none_zero.append([])
        for t, team in enumerate(game):
            none_zero[g].append([])
            if granularity == "Low":
                num_of_players = len(grouped_data[g][t])
                none_zero[g][t] = [num_of_players]
                for m in range(max_length[0] - num_of_players):
                    grouped_data[g][t].append(0)
            else:
                for s, stat in enumerate(team):
                    # print(g,t,s)
                    num_of_players = len(grouped_data[g][t][s])
                    for m in range(max_length[s] - num_of_players):
                        grouped_data[g][t][s].append([stat / num_of_players for stat in average_player[g][t][s]])

                # print(len(grouped_data[g][t][s]))
                # pass
        # for j in range(max_length - len(grouped_data[i][0])):
        #    grouped_data[i][0].append([0] * stat_len)
        # for j in range(max_length - len(grouped_data[i][1])):
        #    grouped_data[i][1].append([0] * stat_len)
        # grouped_data[i][0] += [0] * (max_length - len(grouped_data[i][0]))
        # grouped_data[i][1] += [0] * (max_length - len(grouped_data[i][1]))
        # if len(grouped_data[i][0]) < max_length or len(grouped_data[i][1]) < max_length:
        #    print(len(grouped_data[i][0]), len(grouped_data[i][1]))
        # print(grouped_data[i][0])
        # print(game, len(grouped_data[i][0]),len(grouped_data[i][1]),len(grouped_data[i][0][29]),len(grouped_data[i][1][29]))
        # print(len(grouped_data), len(grouped_data[i]), len(grouped_data[i][0]),len(grouped_data[i][0]))

    print("CPU padded: ", time.process_time() - start_cpu)
    print("ALL padded: ", time.perf_counter() - start_all)
    start_all = time.perf_counter()
    start_cpu = time.process_time()

    if granularity == "Low":
        stat_len = (max(reindex_player_id.values()), max(reindex_player_id.values()))
    else:
        stat_len = (len(grouped_data[0][0][0][0]), len(grouped_data[0][0][1][0]))
    trn_label = []
    trn_data = []
    trn_game = []
    trn_none_zero = []
    tst_label = []
    tst_data = []
    tst_game = []
    tst_none_zero = []

    for game_num, i in game_numbers.items():
        if int(str(game_num)[:4]) < end_year: #> start_year: #
            if not trn_game:
                trn_data.append([])
                trn_label.append([])
                trn_game.append([])
                trn_none_zero.append([])
            elif int(str(game_num)[:4]) != int(str(trn_game[-1][0])[:4]):
                trn_data.append([])
                trn_label.append([])
                trn_game.append([])
                trn_none_zero.append([])
            #if int(str(game_num)[:4]) == end_year and int(str(game_num)[-4:]) > 1082:
            #    continue
            trn_data[-1].append(grouped_data[i])
            trn_label[-1].append(labels[i] + 1)
            trn_game[-1].append(game_num)
            if granularity == "Low":
                trn_none_zero[-1].append(none_zero[i])
            # print(len(trn_data),len(trn_data[-1]))
        elif int(str(game_num)[:4]) == end_year and int(str(game_num)[-4:]) <= 1082:
        #elif int(str(game_num)[:4]) == start_year:# and int(str(game_num)[-4:]) <= 1082:
            if not tst_game:
                tst_data.append([])
                tst_label.append([])
                tst_game.append([])
                tst_none_zero.append([])
            tst_data[-1].append(grouped_data[i])
            tst_label[-1].append(labels[i] + 1)
            tst_game[-1].append(game_num)
            if granularity == "Low":
                tst_none_zero[-1].append(none_zero[i])
    for i, year in enumerate(trn_game):
        last_game = None
        for game in year:
            if str(game)[:4] != last_game:
                print(game)
            last_game = str(game)[:4]
        print(i, len(year))

    # pokus = np.array(trn_data, dtype=object)
    # print(pokus.shape)
    # skaters = np.array(pokus[:, :, 0].tolist(), dtype=float)
    # goalies = np.array(pokus[:, :, 1].tolist(), dtype=float)
    # print(skaters.shape, goalies.shape)
    tst_dataset = {}
    trn_dataset = {}
    trn_dataset["data"] = trn_data
    trn_dataset["labels"] = trn_label
    if granularity == "Low":
        trn_dataset["none_zero"] = trn_none_zero
    trn_loader = Loader(trn_dataset, window_size, test=False)
    tst_dataset["data"] = tst_data
    tst_dataset["labels"] = tst_label
    if granularity == "Low":
        tst_dataset["none_zero"] = tst_none_zero
    tst_loader = Loader(tst_dataset, None, test=True)
    print("CPU ready: ", time.process_time() - start_cpu)
    print("ALL ready: ", time.perf_counter() - start_all)
    start_all = time.perf_counter()
    start_cpu = time.process_time()
    return trn_loader, tst_loader, stat_len


def load_team_data(num_of_train_year, window_size, end_year, granularity, database_location, run_location, file_name):
    if database_location == "local":
        con = psycopg2_connection_to_database()
        table_name = ("cumulative_team_stats", "game_stats")
    elif database_location == "remote":
        con, server = psycopg2_connection_to_remote_database(run_location)
        table_name = ("cumulative_team_stats", "game_stats")
    column_names = get_column_names(table_name[0], database_location, run_location)
    print(column_names)
    game_numbers = []
    # print(num_of_train_year)
    start_year = end_year - num_of_train_year
    # print(start_year)
    if start_year <= 2004:
        start_year -= 1  # -1 protoze v roce 2004 byla vyluka a nehral se jediny zapas
    with con.cursor() as cur:
        cur.execute('SELECT g."teamIdHome",g."teamIdAway",g."goalsHome",g."goalsAway",%s FROM %s AS t \
                     JOIN %s AS g ON t."gameId" = g."gameId" \
                     WHERE t."gameId" >= %s AND t."gameId" < %s \
                     ORDER BY t."timeDate" ASC, t."gameId" ASC'
                    % (",".join(f't."{name}"' for name in column_names), table_name[0], table_name[1],
                       int(str(start_year) + "020000"), int(str(end_year + 1) + "020000")))
        data = cur.fetchall()
    con.close()
    if database_location == "remote":
        server.stop()
    shutil.move(os.path.join("scripts", "loading", file_name), os.path.join("scripts", "loaded", file_name))
    # print(len(data), len(data[0]))
    print(column_names)
    print(data[0])
    game_idx = column_names.index('gameId') + 4
    home_team_idx = column_names.index('teamId0') + 4
    away_team_idx = column_names.index('teamId1') + 4
    home_total_idx = column_names.index('goalsForAllTimeAverageHome0') + 4
    away_total_idx = column_names.index('goalsForAllTimeAverageHome1') + 4
    grouped_data = []
    labels = []

    for item in data:
        game_numbers.append(item[game_idx])
        labels.append(np.sign(item[2] - item[3]))
        if granularity == "Low":
            grouped_data.append(list(item[:2]))
        elif granularity == "Medium":
            grouped_data.append(list(item[home_team_idx + 1:home_total_idx] + item[away_team_idx + 1:away_total_idx]))
            # print(len(grouped_data[-1]))
            # print(grouped_data[-1])
        elif granularity == "High":
            #test = list(item[home_team_idx + 1:away_team_idx] + item[away_team_idx + 1:])
            #test.insert(0,int(str(item[game_idx])[2:4]+str(item[game_idx])[-4:]))
            grouped_data.append(list(item[home_team_idx + 1:away_team_idx] + item[away_team_idx + 1:]))
            # print(len(grouped_data[-1]))
            # print(grouped_data[-1])
    # print(np.unique(np.array(grouped_data).flatten()))
    print(grouped_data[0])
    if granularity == "Low":
        stat_len = np.amax(np.array(grouped_data)) + 1
        # print(stat_len)
    else:
        stat_len = len(grouped_data[0])
    trn_label = []
    trn_data = []
    trn_game = []
    tst_label = []
    tst_data = []
    tst_game = []
    for i, game_num in enumerate(game_numbers):
        if int(str(game_num)[:4]) < end_year:
            if not trn_game:
                trn_data.append([])
                trn_label.append([])
                trn_game.append([])
            elif int(str(game_num)[:4]) != int(str(trn_game[-1][0])[:4]):
                trn_data.append([])
                trn_label.append([])
                trn_game.append([])
            trn_data[-1].append(grouped_data[i])
            trn_label[-1].append(labels[i] + 1)
            trn_game[-1].append(game_num)
            # print(len(trn_data),len(trn_data[-1]))
        elif int(str(game_num)[:4]) == end_year and int(str(game_num)[-4:]) <= 1082:
            if not tst_game:
                tst_data.append([])
                tst_label.append([])
                tst_game.append([])
            tst_data[-1].append(grouped_data[i])
            tst_label[-1].append(labels[i] + 1)
            tst_game[-1].append(game_num)

    tst_dataset = {}
    trn_dataset = {}
    trn_dataset["data"] = trn_data
    trn_dataset["labels"] = trn_label
    # print(len(trn_data), len(trn_data[0]))
    trn_loader = Loader(trn_dataset, window_size, test=False)
    tst_dataset["data"] = tst_data
    tst_dataset["labels"] = tst_label
    # print(len(tst_data), len(tst_data[0]))
    tst_loader = Loader(tst_dataset, None, test=True)
    """
    for i in range(3):
        print("epoch",i)
        for (trn, val) in trn_loader:
            print(trn['data'].shape)
            print(val['data'].shape)
    """
    # trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=False)
    # tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=batch_size)
    return trn_loader, tst_loader, stat_len


class Loader():
    def __init__(self, dataset, sliding_window, test):
        self.test = test
        # self.data = [np.array(year, dtype='f4') for year in dataset["data"]]
        # print(len(dataset["data"]),len(dataset["data"][0]),len(dataset["data"][0][0]),len(dataset["data"][0][0][0]),len(dataset["data"][0][0][0][1]))
        try:
            if "none_zero" in dataset:
                self.none_zero = [np.array(year, dtype='l') for year in dataset["none_zero"]]
                self.data = [np.array(year, dtype=np.int64) for year in dataset["data"]]
            else:
                self.data = [np.array(year, dtype='f4') for year in dataset["data"]]
                self.none_zero = None
        except:

            for year in dataset["data"]:
                np_year = np.array(year, dtype=object)
                skater_year = np_year[:, :, 0].tolist()
                goalie_year = np_year[:, :, 1].tolist()
            self.data = {"skaters": [np.array(np.array(year, dtype=object)[:, :, 0].tolist(), dtype='f4') for year in
                                     dataset["data"]],
                         "goalies": [np.array(np.array(year, dtype=object)[:, :, 1].tolist(), dtype='f4') for year in
                                     dataset["data"]]}

        # print(self.data[0].shape)
        # print(type(dataset["data"][0][0]))
        self.labels = dataset["labels"]
        # print(len(self.labels))
        self.sliding_window = sliding_window
        self.index = 0
        self.window = bool(sliding_window)

    def __len__(self):
        if self.window:
            return len(self.labels) - self.sliding_window
        else:
            return len(self.labels) - 1

    def __iter__(self):
        self.index = 0
        if not self.window:
            self.sliding_window = 0
        return self

    def __next__(self):
        if not self.test:
            trn_batch = {}
            val_batch = {}
            if not self.window:
                self.sliding_window += 1
            if self.index + self.sliding_window + 1 > len(self.labels):
                raise StopIteration
            for i in range(self.sliding_window + 1): #,-1,-1):
                # print("index", i, "sliding window", self.sliding_window)
                # print(trn_batch)
                if not trn_batch:
                    trn_batch["labels"] = torch.tensor(self.labels[self.index + i], dtype=torch.long)
                    try:
                        trn_batch["data"] = torch.from_numpy(self.data[self.index + i])
                        if self.none_zero is not None:
                            trn_batch["none_zero"] = torch.from_numpy(self.none_zero[self.index + i])
                    except:
                        trn_batch["data"] = {"skaters": torch.from_numpy(self.data["skaters"][self.index + i]),
                                             "goalies": torch.from_numpy(self.data["goalies"][self.index + i])}
                elif i < self.sliding_window/2: # i != 0:
                    print("trn", i)
                    trn_batch["labels"] = torch.cat(
                        (trn_batch["labels"], torch.tensor(self.labels[self.index + i], dtype=torch.long)), 0)
                    try:
                        trn_batch["data"] = torch.cat((trn_batch["data"], torch.from_numpy(self.data[self.index + i])),
                                                      0)
                        if self.none_zero is not None:
                            trn_batch["none_zero"] = torch.cat(
                                (trn_batch["none_zero"], torch.from_numpy(self.none_zero[self.index + i])), 0)
                    except:
                        trn_batch["data"]["skaters"] = torch.cat(
                            (trn_batch["data"]["skaters"], torch.from_numpy(self.data["skaters"][self.index + i])), 0)
                        trn_batch["data"]["goalies"] = torch.cat(
                            (trn_batch["data"]["goalies"], torch.from_numpy(self.data["goalies"][self.index + i])), 0)
                else:
                    print("val", i)
                    if not val_batch:
                        val_batch["labels"] = torch.tensor(self.labels[self.index + i], dtype=torch.long)
                        try:
                            val_batch["data"] = torch.from_numpy(self.data[self.index + i])
                            if self.none_zero is not None:
                                val_batch["none_zero"] = torch.from_numpy(self.none_zero[self.index + i])
                        except:
                            val_batch["data"] = {"skaters": torch.from_numpy(self.data["skaters"][self.index + i]),
                                                 "goalies": torch.from_numpy(self.data["goalies"][self.index + i])}
                    else:
                        val_batch["labels"] = torch.cat(
                            (val_batch["labels"], torch.tensor(self.labels[self.index + i], dtype=torch.long)), 0)
                        try:
                            val_batch["data"] = torch.cat(
                                (val_batch["data"], torch.from_numpy(self.data[self.index + i])),
                                0)
                            if self.none_zero is not None:
                                val_batch["none_zero"] = torch.cat(
                                    (val_batch["none_zero"], torch.from_numpy(self.none_zero[self.index + i])), 0)
                        except:
                            val_batch["data"]["skaters"] = torch.cat(
                                (val_batch["data"]["skaters"], torch.from_numpy(self.data["skaters"][self.index + i])),
                                0)
                            val_batch["data"]["goalies"] = torch.cat(
                                (val_batch["data"]["goalies"], torch.from_numpy(self.data["goalies"][self.index + i])),
                                0)

                # print(val_batch["data"].shape)
                # print(trn_batch['data'].shape)
            if self.window:
                self.index += 1
            if val_batch:
                return (trn_batch, val_batch)
            else:
                return (trn_batch)
        else:
            trn_batch = {}
            if self.index == len(self.labels):
                raise StopIteration
            trn_batch["labels"] = torch.tensor(self.labels[0], dtype=torch.long)
            try:
                trn_batch["data"] = torch.from_numpy(self.data[0])
                if self.none_zero is not None:
                    trn_batch["none_zero"] = torch.from_numpy(self.none_zero[0])
            except:
                trn_batch["data"] = {"skaters": torch.from_numpy(self.data["skaters"][0]),
                                     "goalies": torch.from_numpy(self.data["goalies"][0])}
            self.index += 1
            return (trn_batch)


"""
class Loader():
    def __init__(self, dataset, sliding_window, start_year, num_of_val_year, epochs):
        self.dataset = dataset
        self.window = sliding_window
        self.start_year = start_year
        print(start_year)
        print(dataset[0]["game"])
        self.unique_years = []
        last = 0
        for i in dataset:
            if int(str(i["game"])[:4]) != last:
                self.unique_years.append(int(str(i["game"])[:4]))
                last = int(str(i["game"])[:4])
        self.num_of_unique_years = len(self.unique_years)
        self.num_of_val_year = num_of_val_year
        self.num_of_epochs = epochs
        self.num_of_slides = self.num_of_unique_years - (self.window + self.num_of_val_year) + 1
        self.window_years = self.unique_years[self.cur_slide:self.cur_slide+self.window+self.num_of_val_year]
        self.window_index = 0
        print(self.window+self.num_of_val_year, self.num_of_unique_years,self.num_of_slides,self.num_of_epochs//self.num_of_slides)
        self.cur_epoch = 0
        self.cur_slide = 0
        self.trn_batch = {}
        self.val_batch = {}
        self.cur_year = self.window_years[self.window_index]

    def __iter__(self):
        self.cur_epoch += 1
        if self.cur_epoch % (self.num_of_epochs//self.num_of_slides) == 0:
            self.cur_slide += 1
            self.window_years = self.unique_years[self.cur_slide:self.cur_slide + self.window + self.num_of_val_year]
            print("changin")
        return self

    def __next__(self):
        for match in self.dataset:
            #print(self.cur_slide,self.window,self.unique_years[self.cur_slide:self.cur_slide+self.window])
            if int(str(match["game"])[:4]) > self.cur_year: #and int(str(match["game"])[:4]) <= window_years[-1]:
                print(batch["data"].size(),batch["labels"].size())
                self.window_index += 1
                self.cur_year = self.window_years[self.window_index]

            if int(str(match["game"])[:4]) == self.cur_year:
                if not batch:
                    batch["labels"] = torch.tensor([match["labels"]])
                    batch["data"] = torch.unsqueeze(torch.from_numpy(match["data"]), 0)
                else:
                    batch["labels"] = torch.cat((batch["labels"], torch.tensor([match["labels"]])), 0)
                    batch["data"] = torch.cat((batch["data"], torch.unsqueeze(torch.from_numpy(match["data"]), 0)), 0)
                #print(batch["labels"].size(),batch["data"].size())
            elif int(str(match["game"])[:4]) > window_years[-1]:
                raise StopIteration
"""


class Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, data, game):
        self.labels = np.array(labels)
        self.data = np.array(data)
        self.game = np.array(game)
        # print(self.labels.shape)
        # print(self.data.shape)
        """
        self.start_year = START
        self.end_year = END
        self.column_names = get_column_names('cumulative_player_stats_4')
        self.game_numbers = []
        #print(self.column_names)
        with con.cursor() as cur:
            cur.execute('SELECT %s FROM cumulative_player_stats_4 AS p \
                         JOIN game_stats AS g ON p.game = g."gameId" \
                         WHERE p.game >= %s AND p.game < %s \
                         ORDER BY g."timeDate" ASC, p.game ASC, p.side ASC, p.player ASC'
                         % (",".join(f'p."{name}"' for name in self.column_names), int(str(self.start_year) + "020000"), int(str(self.end_year + 1) + "020000")) )
            self.data = cur.fetchall()
        self.grouped_data = []
        self.labels = []
        game_idx = self.column_names.index('game')
        team_idx = self.column_names.index('team')
        player_idx = self.column_names.index('player')
        result_idx = self.column_names.index('results')
        side_idx = self.column_names.index('side')
        self.max_length = 0
        for item in self.data:
            if item[game_idx] not in self.game_numbers:
                self.game_numbers.append(item[game_idx])
                if item[side_idx] == 0:
                    self.labels.append(item[result_idx])
                else:
                    self.labels.append(-item[result_idx])
                self.grouped_data.append([[], []])
            if item[side_idx] == 0:
                self.grouped_data[-1][0] += list(item[result_idx + 1:])
            else:
                self.grouped_data[-1][1] += list(item[result_idx + 1:])
            if self.max_length < len(self.grouped_data[-1][0]):
                self.max_length = len(self.grouped_data[-1][0])
            elif self.max_length < len(self.grouped_data[-1][1]):
                self.max_length = len(self.grouped_data[-1][1])
        #print(self.max_length)
        for i, game in enumerate(self.game_numbers):
            #print(game, self.labels[i], len(self.grouped_data[i][0]), len(self.grouped_data[i][1]))
            self.grouped_data[i][0] += [0] * (self.max_length - len(self.grouped_data[i][0]))
            self.grouped_data[i][1] += [0] * (self.max_length - len(self.grouped_data[i][1]))
            if len(self.grouped_data[i][0]) < self.max_length or len(self.grouped_data[i][1]) < self.max_length:
                print(len(self.grouped_data[i][0]),len(self.grouped_data[i][1]))
        #print(len(self.grouped_data),len(self.grouped_data[0]),len(self.grouped_data[0][0]))
        self.grouped_data = np.array(self.grouped_data)
        #print(self.grouped_data.shape)
        self.labels = np.array(self.labels)
        #print(self.labels.shape)
        """
        """    
            if item[game_idx] not in self.game_numbers:
                self.game_numbers.append(item[game_idx])
            if item[game_idx] not in self.grouped_data:
                self.grouped_data[item[game_idx]] = {}
            if item[team_idx] not in self.grouped_data[item[game_idx]]:
                self.grouped_data[item[game_idx]][item[team_idx]] = []
            self.grouped_data[item[game_idx]][item[team_idx]] += list(item[result_idx+1:])
            if item[game_idx] not in self.labels:
                self.labels[item[game_idx]] = {'results': {}, 'side': {}}
            if item[team_idx] not in self.labels[item[game_idx]]['results']:
                self.labels[item[game_idx]]['results'][item[team_idx]] = item[result_idx]
                self.labels[item[game_idx]]['side'][item[team_idx]] = item[side_idx]
        for name in self.column_names:
            print(name, self.data[0][self.column_names.index(name)])
        for game in self.grouped_data:
            for team in self.grouped_data[game]:
                #print(len(self.grouped_data[game][team]))
                if len(self.grouped_data[game][team]) > self.max_length:
                    self.max_length = len(self.grouped_data[game][team])
            
        print(self.grouped_data[item[game_idx]])
        print(self.labels[item[game_idx]])
        #print(self.max_length)
        """
        # print(self.game_numbers)
        # values = len(np.unique(list(zip(*self.data))[1]))
        # print(len(self.game_numbers),len(values))
        """
            data_array = []
        for teamId in data:
            data_vector = [0] * 54
            #print(teamId[1],teamId[2])
            data_vector[teamId[1]-1] = 1
            data_vector[teamId[2]-1] = 1
            data_array.append(data_vector)
        #print(data_array)
        #self.data = [np.array(teamId[1:3]) for teamId in data]
        self.data = data_array
        #print(data[0])
        #print(self.data[0])
        goal_diff = [goals[3]-goals[4] for goals in data]
        self.labels = np.sign(goal_diff)+1
        weights = [len(goal_diff) / np.count_nonzero(self.labels == 0), len(goal_diff) / np.count_nonzero(self.labels == 1),
                   len(goal_diff) / np.count_nonzero(self.labels == 2)]
        print(weights)

        #print(self.labels)
        """

    def __getitem__(self, i):
        """
        return_dict = {'labels': [], 'data': []}
        game_number = self.game_numbers[i]
        teams = list(self.labels[game_number]['side'].keys())
        if self.labels[game_number]['side'][teams[0]] == 0:
            home_team = teams[0]
            away_team = teams[1]
        elif self.labels[game_number]['side'][teams[1]] == 0:
            home_team = teams[1]
            away_team = teams[0]
        else:
            print('something is wrong')
        return_dict['labels'] = self.labels[game_number]['results'][home_team]

        return_dict['data'] = self.grouped_data[game_number][home_team] + [0]*(self.max_length-len(self.grouped_data[game_number][home_team])) + self.grouped_data[game_number][away_team] + [0]*(self.max_length-len(self.grouped_data[game_number][away_team]))
        """
        return {
            'game': self.game[i],
            'labels': self.labels[i].astype('i8'),
            # torch wants labels to be of type LongTensor, in order to compute losses
            'data': self.data[i].astype('f4')  # First retype to float32 (default dtype for torch)
        }

    def __len__(self):
        # return len(np.unique(list(zip(*self.data))[1]))
        return self.labels.size


if __name__ == '__main__':
    goalie_stats = ['timeOnIce', 'pim', 'shortHandedShotsAgainst', 'powerPlayShotsAgainst', 'evenShotsAgainst', 'saves',
                    'shots', 'assists', 'goals', 'shortHandedSaves', 'powerPlaySaves', 'evenSaves', 'goalsPerMinute',
                    'shotsPerMinute', 'assistsPerMinute', 'pimPerMinute', 'goalsPerGame', 'shotsPerGame',
                    'assistsPerGame',
                    'pimPerGame', 'timeOnIcePerGame', 'savesPerMinute', 'savesPerGame', 'savesPercentage',
                    'powerPlaySavesPerGame', 'powerPlaySavesPercentage', 'shortHandedSavesPerGame',
                    'shortHandedSavesPercentage', 'evenSavesPerGame', 'evenSavesPercentage']
    skater_stats = ['timeOnIce', 'takeaways', 'faceoffTaken', 'shortHandedAssists', 'faceOffWins', 'powerPlayTimeOnIce',
                    'shots', 'assists', 'goals', 'plusMinus', 'hits', 'shortHandedGoals', 'evenTimeOnIce', 'blocked',
                    'goalsPerMinute', 'shotsPerMinute', 'assistsPerMinute', 'powerPlayGoals', 'powerPlayAssists',
                    'goalsPerGame', 'shotsPerGame', 'assistsPerGame', 'timeOnIcePerGame', 'shortHandedTimeOnIce',
                    'giveaways', 'hitsPerMinute', 'goalsPercentage', 'takeawaysPerMinute', 'giveawaysPerMinute',
                    'blockedPerMinute', 'powerPlayGoalsPerMinute', 'powerPlayAssistsPerMinute',
                    'shortHandedGoalsPerMinute',
                    'shortHandedAssistsPerMinute', 'evenGoalsPerMinute', 'evenAssistsPerMinute', 'hitsPerGame',
                    'takeawaysPerGame', 'giveawaysPerGame', 'blockedPerGame', 'powerPlayGoalsPerGame',
                    'powerPlayAssistsPerGame', 'shortHandedGoalsPerGame', 'shortHandedAssistsPerGame',
                    'evenGoalsPerGame',
                    'evenAssistsPerGame', 'faceoffPercentage', 'powerPlayTimeOnIcePerGame',
                    'shortHandedTimeOnIcePerGame',
                    'teamGoalsPercentage', 'teamShotsPercentage', 'teamBlockedPercentage', 'teamTakeawaysPercentage',
                    'teamHitsPercentage']
    col_names = get_column_names("cumulative_player_stats_6", "local", "local")

    print(col_names)
    # print(np.array(col_names)[skater_stats_index])
    # print(np.array(col_names)[np.where(np.isin(col_names,goalie_stats+skater_stats,invert=True))])

    trn, tst, stat_len = load_player_data(3, 2, 2019, "High", "local", "local", "b.txt")

    print(len(trn))
    for i, (trn_b, val_b) in enumerate(trn):
        # print((trn_b["data"]))
        print((trn_b["data"]["skaters"].size()))
        print((trn_b["data"]["goalies"].size()))
        print((val_b["data"]["skaters"].size()))
        print((val_b["data"]["goalies"].size()))
        print(trn_b["data"]["skaters"][0][0][0][:6])
        print(val_b["data"]["skaters"][0][0][0][:6])
        print(trn_b["data"]["skaters"][4960][0][0][:6])
        print(val_b["data"]["skaters"][1270][0][0][:6])
        # print(len(trn_b["data"]))
        # print(type(trn_b["data"]))
        # print(type(trn_b))

        pass
    """
    data = Dataset(psycopg2_connection_to_database(),2000, 2002)
    print(len(data))
    for i in range(len(data)):
        print(data[i])
        if i == len(data)-1:
            break
        if len(data[i]) != len(data[i+1]):
            print(len(data[i]), len(data[i+1]))
        else:
            print(i,'ok')
    """
