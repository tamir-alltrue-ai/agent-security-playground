import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

TRACES_DIR = os.path.join(ROOT_DIR, "traces")

PROXY_LOG_DIR = os.path.join(ROOT_DIR, "proxy_logs")