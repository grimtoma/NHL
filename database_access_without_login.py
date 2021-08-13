import psycopg2
from sqlalchemy import create_engine
import sshtunnel

def create_ssh_tunnel(run_location="server"):
    if run_location == "local":
      PATH_TO_PK = '../../.ssh/id_rsa'
    elif run_location == "server":
      PATH_TO_PK = './.ssh/id_rsa'
    PK_PW = ''
    FRONTEND_IP = '78.128.250.175'
    DB_IP = '172.16.1.180'
    USERNAME = ''
    server = sshtunnel.SSHTunnelForwarder(
            (FRONTEND_IP, 22),
            ssh_username=USERNAME,
            ssh_pkey=PATH_TO_PK,
            ssh_private_key_password=PK_PW,
            remote_bind_address=(DB_IP, 5432),
            )
    server.start()
    print("SSH tunnel successfully created")
    return server


def close_ssh_tunnel(server):
    server.stop()


def sqlalchemy_connection_to_remote_database(run_location):
    server = create_ssh_tunnel(run_location)
    BIND_PORT = server.local_bind_port
    DB = 'students'
    DB_USER = ''
    DB_PW = ''
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PW}@127.0.0.1:{BIND_PORT}/{DB}')
    print("Engine successfully created ")
    return engine, server

def psycopg2_connection_to_remote_database(run_location):
    server = create_ssh_tunnel(run_location)
    BIND_PORT = server.local_bind_port
    DB = 'students'
    DB_USER = ''
    DB_PW = ''
    con = psycopg2.connect(host="127.0.0.1",
                           database=DB,
                           user=DB_USER,
                           password=DB_PW,
                           port=BIND_PORT,
                           options='-c search_path=nhl')
    print("Connection successfully created")
    return con, server


def sqlalchemy_connection_to_database(host="localhost", database="nhl", user="postgres", password="password"):
    param_dic = {"host": host,
                 "database": database,
                 "user": user,
                 "password": password}
    connect = "postgresql+psycopg2://%s:%s@%s:5432/%s" % (param_dic['user'], param_dic['password'], param_dic['host'], param_dic['database'])
    return create_engine(connect)


def psycopg2_connection_to_database(host="localhost", database="nhl", user="postgres", password="password"):
    con = psycopg2.connect(host=host,
                           database=database,
                           user=user,
                           password=password)
    return con
