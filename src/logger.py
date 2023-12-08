import logging
import os
from datetime import datetime

# log file name
LOG_FILE  = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# to get current working directory and join logs/log file names
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

# make directory if already exists append to it
# os.makedirs(logs_path,exist_ok=True)

# to get log file path
# LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)
LOG_FILE_PATH = logs_path

logging.basicConfig(
    # file name
    filename=LOG_FILE_PATH,
    
    # format of the log file (content format in log file)
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    
    # level is what level you want information like debuging warning critical etc
    level=logging.INFO,
)