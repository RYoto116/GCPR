import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from reckit.model.abstract_recommender import AbstractRecommender
from reckit.rand import randint_choice
from data import PairwiseSamplerV2
from time import time
from reckit.util import l2_loss
from reckit.util import timer
import warnings
from reckit.util import sp_mat_to_sp_tensor
from .GCPR import LightGCN

class SGL(AbstractRecommender):
    def __init__(self, config):
        super(SGL, self).__init__(config)
        
        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]
        
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

        self.n_layers = config["n_layers"]

        self.ssl_aug_type = config["aug_type"]          # 数据增强方式
        assert self.ssl_aug_type in ['nd', 'ed', 'rw']
        self.ssl_reg = config["ssl_reg"]                # 正则化系数
        self.ssl_ratio = config["ssl_ratio"]            # 不同k的采样比率
        self.ssl_temp = config["ssl_temp"]                # 温度系数
        self.ssl_mode = config["ssl_mode"]              # 对比模式

        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)
        self.model_str = '#layers=%d-reg=%.0e' % (self.n_layers, self.reg) + \
                         '/1ratio=%.f-mode=%s-temp=%.2f-reg=%.0e' \
                         % (self.ssl_ratio, self.ssl_mode, self.ssl_temp, self.ssl_reg)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn =LightGCN(self.num_users, self.num_items, self.embedding_size, adj_matrix, self.n_layers).to(self.device)
        self.lightgcn.reset_parameters(init_method=self.param_init)

        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        warnings.filterwarnings('ignore')
        
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
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            # Loss = CPR Loss + InfoNCE Loss + Reg Loss
            total_loss, total_sup_loss, total_reg_loss = 0.0, 0.0, 0.0
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

            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, ssl_logits_user, ssl_logits_item, _ = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)
                
                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss
                total_loss += loss
                total_sup_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch, 
                total_loss/self.num_ratings,
                total_sup_loss / self.num_ratings,
                (total_loss - total_sup_loss - total_reg_loss) / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time()-training_start_time,)
            )
            if epoch % self.verbose == 0 and epoch > self.config.start_testing_epoch:
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
        buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
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
