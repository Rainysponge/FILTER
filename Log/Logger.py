import logging
import time


def getTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class Log():

    def __init__(self, className, parse=None):
        if parse is None:
            parse = {"log_save_path": "./log_save/log.txt"}
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(process)d \n\t %('
                                                       'message)s')
        self.className = className
        self.savePath = parse["log_save_path"]  # 来自配置文件
        self.logger = logging.getLogger(className)
        self.logger.setLevel(level=logging.INFO)
        self.handler = logging.FileHandler(self.savePath)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
        self.handler.setFormatter(self.formatter)

        self.logger.addHandler(self.handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.info(message)

    def critical(self, message):
        self.logger.info(message)

    def save(self):
        pass

    def change_save_path(self, path):
        self.savePath = path
        self.handler = logging.FileHandler(path)
