import logging
import os

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG:
            record.levelname = '\033[1;34m%s\033[0m' % record.levelname
        elif record.levelno == logging.INFO:
            record.levelname = '\033[1;32m%s\033[0m' % record.levelname
        return super().format(record)
    
def set_logging():
    log_level = os.getenv('LOG_LEVEL')
    if log_level == 'DEBUG':
        logging_level = logging.DEBUG
    if log_level == 'INFO':
        logging_level = logging.INFO 
    else:
        logging_level = logging.DEBUG    
    # logging.basicConfig(level=logging_level, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    if not logging.getLogger().handlers:
        console = logging.StreamHandler()
        console.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(console)
        logging.getLogger().setLevel(logging_level)