from data import Dataset
from reckit import Logger, Evaluator

class AbstractRecommender(object):
    def __init__(self, config):
        # 构建三个数据集
        self.dataset = Dataset(config.data_dir, config.daset, config.sep, config.file_column)
        self.logger = self._create_logger(config, self.dataset)
        user_train_dict = self.dataset.train_data.to_user_dict()
        user_test_dict = self.dataset.test_data.to_user_dict()
        self.evaluator = Evaluator(self.dataset, user_train_dict, user_test_dict, metric=config.metric, top_k=config.top_k, batch_size=config.batch_size, num_thread=config.num_thread)

    def _create_logger(self, config, dataset):
        return Logger()
