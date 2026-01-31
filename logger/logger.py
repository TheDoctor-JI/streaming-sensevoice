import logging
import yaml
import os
from inspect import getsourcefile
from datetime import datetime


#HARDCODE log dir
logging_configs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './logging_configs.yaml')

def get_root_log_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'logs')



def get_log_file_name(logger_name: str, date_format_str: str) -> str:
    date_string = datetime.now().strftime(date_format_str)
    return date_string + '_' + logger_name + '.log'



#Create Logger
def setup_logger(
    logger_name: str, 
    file_log_level: str = 'INFO', 
    terminal_log_level: str = 'INFO', 
    logger_config_path: str = logging_configs_path
    ):

    #Format string for log line and log date
    with open(logger_config_path, "r", encoding='utf-8') as config_file:
        logger_config = yaml.safe_load(config_file)
    file_line_format_str = logger_config['file_line_format']
    file_date_format_str = logger_config['file_date_format']
    terminal_line_format_str = logger_config['terminal_line_format']
    terminal_date_format_str = logger_config['terminal_date_format']

    #Map log level to int values
    file_log_level = logger_config['log_levels'][file_log_level]
    terminal_log_level = logger_config['log_levels'][terminal_log_level]

    # if the log folder does not exist, create it
    log_dir = os.path.join(get_root_log_path(),logger_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = get_log_file_name(logger_name, file_date_format_str)
    log_file_path = os.path.join(log_dir, log_file_name)
    # log to terminal that the folder is created
    f = open(log_file_path, "a")
    f.close()
    print("Log file %s has been created." % log_file_path )


    #Format of log lines   
    file_log_formatter = logging.Formatter(file_line_format_str, datefmt = file_date_format_str)
    terminal_log_formatter = logging.Formatter(terminal_line_format_str, datefmt = terminal_date_format_str)


    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(file_log_formatter)
    file_handler.setLevel(file_log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(terminal_log_formatter)
    stream_handler.setLevel(terminal_log_level)

    #Create logger
    logger = logging.getLogger(logger_name)
    # setattr(logger, 'log_file_name', log_file_name)

    #To prevent duplication
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.propagate = False

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel( min(file_log_level,terminal_log_level) )


    return logger