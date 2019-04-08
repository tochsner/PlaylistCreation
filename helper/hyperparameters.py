"""
A class for storing and saving hyperparameters.
"""

from helper.logger import Logger
from datetime import datetime

class HyperparameterLogger:
    def __init__(self, dict, name="", detailed_name=""):
        self.dict = dict
        self.logger = Logger(name)
        self.detailed_logger = Logger(detailed_name)
    
    def log(self, result):
        self.logger.log(";".join(map(str, [datetime.now(), self.dict, result])))

    def log_all_epochs(self, epochs):
        self.detailed_logger.log("***********")
        self.detailed_logger.log(";".join(map(str, [datetime.now(), self.dict])))

        for epoch in epochs:
            self.detailed_logger.log(epoch)