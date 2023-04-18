import numpy as np
from reckit.dataiterator import DataIterator
from reckit.cython import is_ndarray, float_type
from reckit.cython import eval_score_matrix

metric_dict = {"Precision": 1, "Recall": 2, "MAP": 3, "NDCG": 4, "MRR": 5, "ARP": 6}
re_metric_dict = {value: key for key, value in metric_dict.items()}

# Evaluator for item ranking task.
# five configurable metrics: `Precision`,`Recall`,`MAP`,`NDCG`, `MRR`
class Evaluator(object):
    def __init__(self, dataset, user_train_dict, user_test_dict, metric=None, top_k=50, batch_size=1024, num_thread=8):
        super(Evaluator, self).__init__()
        if metric is None:
            metric = ["Precision", "Recall", "MAP", "NDCG", "MRR"]
        elif isinstance(metric, str):
            metric = [metric]
        elif isinstance(metric, (set, tuple, list)):
            pass
        else:
            raise TypeError("The type of 'metric' (%s) is invalid!" % metric.__class__.__name__)

        for m in metric:
            if m not in metric_dict:
                raise ValueError("There is not the metric named '%s'!" % metric)

        self.dataset = dataset
        self.item_degrees = dataset.item_degrees
        self.user_pos_train = user_train_dict if user_train_dict is not None else dict()
        self.user_pos_test = user_test_dict if user_test_dict is not None else dict()

        self.metric_len = len(metric)
        self.metrics = [metric_dict[m] for m in metric]
        self.num_thread = num_thread
        self.batch_size = batch_size
        self.max_top = top_k if isinstance(top_k, int) else max(top_k)
        if isinstance(top_k, int):
            self.top_show = np.arange(top_k) + 1
        else:
            self.top_show = np.sort(top_k)

    def metrics_info(self):
        """Get all metrics information.
        """
        metrics_show = ['\t'.join([("%s@" % re_metric_dict[metric] + str(k)).ljust(12) for k in self.top_show])
                        for metric in self.metrics]
        metric = '\t'.join(metrics_show)
        return "metrics:\t%s" % metric

    def evaluate(self, model, test_users=None):
        """Evaluate model
        Args:
            model: 必须有方法 `predict(self, users)`，其中`users`是user列表, 返回值是一个2维向量，包含`user`对所有item的评分
            test_users: Default is None and means test all users in user_pos_test.
        
        return: 
            str, A single-line string consist of all results
        """
        if not hasattr(model, "predict"):
            raise AttributeError("'model' must have attribute 'predict'.")
        test_users = test_users if test_users is not None else list(self.user_pos_test.keys())
        if not isinstance(test_users, (list, tuple, set, np.ndarray)):
            raise TypeError("'test_user' must be a list, tuple, set or numpy array!")

        test_users = DataIterator(test_users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            test_items = [self.user_pos_test[u] for u in batch_users]
            ranking_score = model.predict(batch_users)  # [batch_size, num_items]
            if not is_ndarray(ranking_score, float_type):
                ranking_score = np.array(ranking_score, dtype=float_type)
            # 将训练样本的分数设为-inf
            for idx, user in enumerate(batch_users):
                if user in self.user_pos_train and len(self.user_pos_train[user]) > 0:
                    train_items = self.user_pos_train[user]
                    ranking_score[idx][train_items] = -np.inf
            # result = eval_score_matrix(ranking_score, test_items, self.metrics, self.max_top, self.num_thread)  # [batch_size, topk * num_metrics]
            result = eval_score_matrix(ranking_score, test_items, self.metrics, list(self.item_degrees), self.max_top, self.num_thread)  # [batch_size, topk * num_metrics]
            batch_result.append(result)

        all_user_result = np.concatenate(batch_result, axis=0)  # [num_users, topk * num_metrics]
        final_result = np.mean(all_user_result, axis=0)         # [topk * num_metrics]
        final_result = np.reshape(final_result, newshape=[self.metric_len, self.max_top])  # [num_metrics, topk]
        final_result = final_result[:, self.top_show-1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join([("%.8f" % x).ljust(12) for x in final_result])

        return final_result, buf