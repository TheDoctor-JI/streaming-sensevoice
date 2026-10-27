"""File-backed logging with timestamped directories and inherited artifact paths."""
import logging
import os
import re
import threading
from datetime import datetime, timedelta

import yaml

logging_configs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logging_configs.yaml')
DIRECTORY_DATE_FORMAT = '%Y_%m_%d_%H_%M_%S_%f'
_DIRECTORY_PATTERN = re.compile(r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{6}_(.+)$')
_setup_lock = threading.RLock()


def get_root_log_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')


def log_directory_logger_name(directory):
    """Extract the owner name from a timestamped directory, or return None."""
    match = _DIRECTORY_PATTERN.fullmatch(os.path.basename(os.fspath(directory)))
    return match.group(1) if match else None


def create_log_directory(logger_name, root=None) -> str:
    """Atomically reserve a fresh directory, including on timestamp collisions."""
    if (not logger_name or logger_name in ('.', '..')
            or '/' in logger_name or '\\' in logger_name):
        raise ValueError('logger_name must be a nonempty directory component')
    root = os.path.abspath(os.fspath(root if root is not None else get_root_log_path()))
    os.makedirs(root, exist_ok=True)
    started = datetime.now()
    collision = 0
    while True:
        stamp = (started + timedelta(microseconds=collision)).strftime(DIRECTORY_DATE_FORMAT)
        directory = os.path.join(root, stamp + '_' + logger_name)
        try:
            os.mkdir(directory)
            return directory
        except FileExistsError:
            collision += 1


def find_log_directories(logger_name, root=None):
    """List this owner's timestamped directories in chronological name order."""
    root = os.fspath(root if root is not None else get_root_log_path())
    if not os.path.isdir(root):
        return []
    with os.scandir(root) as entries:
        return sorted(entry.path for entry in entries
                      if entry.is_dir(follow_symlinks=False)
                      and log_directory_logger_name(entry.name) == logger_name)


def get_log_file_path(logger) -> str:
    """Resolve a file handler through the effective logger ancestry."""
    cursor = logging.getLogger(logger) if isinstance(logger, str) else logger
    while cursor is not None:
        files = [handler for handler in cursor.handlers
                 if isinstance(handler, logging.FileHandler)
                 and not getattr(handler, '_closed', False)]
        owned = [handler for handler in files if getattr(handler, '_session_log_owned', False)]
        if owned or files:
            return (owned or files)[0].baseFilename
        if not cursor.propagate:
            break
        cursor = cursor.parent
    raise ValueError('Logger has no effective file handler')


def get_log_directory(logger) -> str:
    return os.path.dirname(get_log_file_path(logger))


def get_log_file_name(logger_name: str, date_format_str: str) -> str:
    return datetime.now().strftime(date_format_str) + '_' + logger_name + '.log'


def setup_logger(logger_name: str, file_log_level: str = 'INFO',
                 terminal_log_level: str = 'INFO', logger_config_path: str = logging_configs_path):
    """Configure owned handlers once; children and repeated setup share a directory."""
    with open(logger_config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    file_level = config['log_levels'][file_log_level]
    terminal_level = config['log_levels'][terminal_log_level]
    file_formatter = logging.Formatter(config['file_line_format'], datefmt=config['file_date_format'])
    terminal_formatter = logging.Formatter(config['terminal_line_format'], datefmt=config['terminal_date_format'])
    root = os.path.abspath(get_root_log_path())
    with _setup_lock:
        logger = logging.getLogger(logger_name)
        owned = [handler for handler in logger.handlers if getattr(handler, '_session_log_owned', False)]
        files = [handler for handler in owned if isinstance(handler, logging.FileHandler)]
        streams = [handler for handler in owned if not isinstance(handler, logging.FileHandler)]
        reusable = (len(files) == len(streams) == 1
                    and getattr(files[0], '_session_log_root', None) == root
                    and not files[0]._closed and not streams[0]._closed
                    and os.path.isdir(os.path.dirname(files[0].baseFilename)))
        if reusable:
            file_handler, stream_handler = files[0], streams[0]
        else:
            directory = create_log_directory(logger_name, root)
            path = os.path.join(directory, get_log_file_name(logger_name, config['file_date_format']))
            file_handler = logging.FileHandler(path, encoding='utf-8')
            stream_handler = logging.StreamHandler()
            for handler in (file_handler, stream_handler):
                handler._session_log_owned = True
                handler._session_log_root = root
            for handler in owned:
                logger.removeHandler(handler)
                handler.close()
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_level)
        stream_handler.setFormatter(terminal_formatter)
        stream_handler.setLevel(terminal_level)
        logger.propagate = False
        logger.setLevel(min(file_level, terminal_level))
        if not reusable:
            print('Log file %s has been created.' % file_handler.baseFilename)
        return logger
