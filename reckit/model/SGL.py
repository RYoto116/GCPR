import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from reckit.model.abstract_recommender import AbstractRecommender
from reckit import OrderedDict
from reckit.rand import randint_choice
from functools import partial
from reckit import CPRSampler
from time import time
from reckit.util import l2_loss, cpr_loss

def inner_product(a, b):
    return torch.sum(a*b, dim=-1)

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

class InitArg(object):
    MEAN = 0.0
    STDDEV = 0.01
    MIN_VAL = -0.05
    MAX_VAL = 0.05

def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_(mean=0, std=1)
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

_initializers = OrderedDict()
_initializers["normal"] = partial(nn.init.normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["truncated_normal"] = partial(truncated_normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["uniform"] = partial(nn.init.uniform_, a=InitArg.MIN_VAL, b=InitArg.MAX_VAL)
_initializers["he_normal"] = nn.init.kaiming_normal_
_initializers["he_uniform"] = nn.init.kaiming_uniform_
_initializers["xavier_normal"] = nn.init.xavier_normal_
_initializers["xavier_uniform"] = nn.init.xavier_uniform_
_initializers["zeros"] = nn.init.zeros_
_initializers["ones"] = nn.init.ones_

def get_initializer(init_method):
    if init_method not in _initializers:
        init_list = ', '.join(_initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    return _initializers[init_method]


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, norm_adj, n_layers=3):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.norm_adj = norm_adj
        self.n_layers = n_layers

        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_size)
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)      # [batch_size, embedding_size]
        item_embs = F.embedding(items, item_embeddings)      # [batch_size, embedding_size]
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)    # [batch_size, embedding_size]
        item_embs1 = F.embedding(items, item_embeddings1)    # [batch_size, embedding_size]
        user_embs2 = F.embedding(users, user_embeddings2)    # [batch_size, embedding_size]
        item_embs2 = F.embedding(items, item_embeddings2)    # [batch_size, embedding_size]

        sup_pos_ratings = inner_product(user_embs, item_embs)        # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)    # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings

        pos_ratings_user = inner_product(user_embs1, user_embs2)     # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)     # [batch_size]
        tot_ratings_user = inner_product(user_embs1, torch.transpose(user_embeddings2, 0, 1))    # [batch_size, num_users]
        tot_ratings_item = inner_product(item_embs1, torch.transpose(item_embeddings2, 0, 1))    # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user
        ssl_logits_item = tot_ratings_item - pos_ratings_item

        return sup_logits, ssl_logits_user, ssl_logits_item

    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch.sparse.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)

            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")

        user_embs = F.embedding(users, self._user_embeddings_final)
        tmp_item_embeddings =self._item_embeddings_final
        ratings = torch.matmul(user_embs, tmp_item_embeddings.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

class SGL(AbstractRecommender):
    def __init__(self, config):
        super(SGL, self).__init__(config)
        # 基类的成员变量包括dataset, logger, evaluator
        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # 通用超参数
        self.reg = config["reg"]
        self.embedding_size = config["embedding_size"]
        self.lr = config["lr"]
        self.learner = config["learner"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.param_init = config["param_init"]

        # GCN超参数
        self.n_layers = config["n_layers"]

        # 对比学习超参数
        self.ssl_aug_type = config["aug_type"]          # 数据增强方式
        assert self.ssl_aug_type in ['nd', 'ed', 'rw']
        self.ssl_reg = config["ssl_reg"]                # 正则化系数
        self.ssl_ratio = config["ssl_ratio"]            # 不同k的采样比率
        self.ssl_temp = config["ssl_temp"]                # 温度系数
        self.ssl_mode = config["ssl_mode"]              # 对比模式

        # 其他超参数
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (self.n_layers, self.reg) + \
                         '/1ratio=%.f-mode=%s-temp=%.2f-reg=%.0e' \
                         % (self.ssl_ratio, self.ssl_mode, self.ssl_temp, self.ssl_reg)
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' \
                                 % (self.dataset_name, self.model_name, self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' \
                            % (self.dataset_name, self.model_name,self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_ratings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn =_LightGCN(self.num_users, self.num_items, self.embedding_size, adj_matrix, self.n_layers).to(self.device)
        self.lightgcn.reset_parameters(init_method=self.param_init)

        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        # 数据增强构建子图
        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                drop_user_idx = randint_choice(self.num_users, size=self.num_users * self.ssl_ratio, replace=False)
                drop_item_idx = randint_choice(self.num_items, size=self.num_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)

                indicator_user[drop_user_idx] = 0.0
                indicator_item[drop_item_idx] = 0.0

                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)

                R = sp.csr_matrix((np.ones_like(users_np, dtype=np.float32), (users_np, items_np)), shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(n_nodes, n_nodes))
            elif aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = users_np[keep_idx]
                item_np = items_np[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        # 原图
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # 计算度数
        degrees = np.array(adj_mat.sum(1))
        d_inv = np.power(degrees, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        # 归一化
        norm_adj = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        sample_ratio, sample_rate, n_thread = self.config["sample_ratio"], self.config["sample_rate"], self.config["num_thread"]
        data_iter = CPRSampler(self.dataset, sample_ratio=sample_ratio, sample_rate=sample_rate, batch_size=self.batch_size, n_thread=n_thread)

        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_cpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            # 构建子图
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
                sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            elif self.ssl_aug_type == 'rw':
                sub_graph1, sub_graph2 = [], []
                for _ in range(self.n_layers):
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph1.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph2.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))

            self.lightgcn.train()  # Sets the module in training mode.

            for bat_users, bat_items in data_iter.sample():
                self.lightgcn(sub_graph1, sub_graph2, bat_users, bat_items, None)

                batch_user_embeds = self.lightgcn.user_embeddings(bat_users)
                batch_pos_item_embeds = self.lightgcn.item_embeddings(bat_items)

                u_splits = torch.split(batch_user_embeds, data_iter.batch_total_sample_sizes, 0)
                i_splits = torch.split(batch_pos_item_embeds, data_iter.batch_total_sample_sizes, 0)
                pos_scores = []
                neg_scores = []

                for idx in range(len(data_iter.batch_total_sample_sizes)):
                    u_list = np.split(u_splits[idx], idx+2, 0)
                    i_list = np.split(i_splits[idx], idx+2, 0)

                    pos_scores.append(torch.reduce_mean([inner_product(u, i) for u, i in zip(u_list, i_list)], axis=0))
                    neg_scores.append(torch.reduce_mean([inner_product(u, i) for u, i in zip(u_list, i_list[1:] + i_list[0])], axis=0))

                pos_scores = torch.concat(pos_scores, axis=0)
                neg_scores = torch.concat(neg_scores, axis=0)

                cpr = cpr_loss(pos_scores, neg_scores, self.batch_size, sample_rate)
                reg_loss = l2_loss(batch_user_embeds, batch_pos_item_embeds)

                loss = cpr + self.reg * reg_loss
                total_loss += loss
                total_cpr_loss += cpr
                total_reg_loss += self.reg * reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f, time: %f]" % (
                epoch,
                total_loss / self.num_ratings,
                total_cpr_loss / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time() - training_start_time,)
            )
            
            if epoch % self.verbose == 0 and epoch > self.config["starting_testing_epoch"]:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d\t%s" % (epoch, result))
                # 找到了更好的参数
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break
        
        self.logger.info("best results@epoch %d\n" % self.best_epoch)
        buf = '\t'.join([("%4.f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()  # 得到self._user_embeddings_final 和 self._item_embeddings_final
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()  # ratings
