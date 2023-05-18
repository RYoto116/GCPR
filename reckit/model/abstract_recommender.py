from data import Dataset
from reckit import Logger, Evaluator
import time
import os

class AbstractRecommender(object):
    def __init__(self, config):
        # 构建三个数据集
        self.dataset = Dataset(config.data_dir, config.dataset, config.sep, config.file_column)
        self.logger = self._create_logger(config, self.dataset)
        user_train_dict = self.dataset.train_data.to_user_dict()
        user_test_dict = self.dataset.test_data.to_user_dict()
        self.evaluator = Evaluator(self.dataset, user_train_dict, user_test_dict, metric=config.metric, top_k=config.top_k, batch_size=config.batch_size, num_thread=config.num_thread)

    def _create_logger(self, config, dataset):
        timestamp = time.time()
        model_name = self.__class__.__name__
        data_name = dataset.data_name
        param_str = f"{config.summarize()}"
        
        run_id = f"{param_str[:150]}_{timestamp:.8f}"
        
        log_dir = os.path.join(config.root_dir + "log", data_name, model_name)
        logger_name = os.path.join(log_dir, run_id + ".log")
        logger = Logger(logger_name)
        
        logger.info(f"my pid: {os.getpid()}")
        logger.info(f"model: {self.__class__.__module__}")
        logger.info(self.dataset)
        logger.info(config)
        
        return logger
