import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from reckit.model.abstract_recommender import AbstractRecommender
from reckit.rand import randint_choice
from reckit import CPRSampler
from time import time
from reckit.util import l2_loss, cpr_loss
from reckit.util import timer
import copy
import warnings
from reckit.util import inner_product
from reckit.util import sp_mat_to_sp_tensor
from reckit.util import get_initializer
 
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, norm_adj, n_layers=3):
        super(LightGCN, self).__init__()
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

        return [user_embeddings, item_embeddings], [user_embeddings1, item_embeddings1], [user_embeddings2, item_embeddings2]


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
        super(LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

class GCPR(AbstractRecommender):
    def __init__(self, config):
        super(GCPR, self).__init__(config)
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
        self.num_thread = config["num_thread"]

        # GCN超参数
        self.n_layers = config["n_layers"]

        # 对比学习超参数
        self.ssl_aug_type = config["aug_type"]          # 数据增强方式
        assert self.ssl_aug_type in ['nd', 'ed', 'rw']
        self.ssl_reg = config["ssl_reg"]                # 正则化系数
        self.ssl_ratio = config["ssl_ratio"]            # 不同k的采样比率
        self.ssl_temp = config["ssl_temp"]                # 温度系数
        self.ssl_mode = config["ssl_mode"]              # 对比模式
        self.sample_ratio = config["sample_ratio"]
        self.sample_rate = config["sample_rate"]

        # 其他超参数
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)
        self.model_str = '#layers=%d-reg=%.0e' % (self.n_layers, self.reg) + \
                         '/1ratio=%.f-mode=%s-temp=%.2f-reg=%.0e' \
                         % (self.ssl_ratio, self.ssl_mode, self.ssl_temp, self.ssl_reg)
                         
        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        data_iter = CPRSampler(self.dataset, sample_ratio=self.sample_ratio, sample_rate=self.sample_rate, batch_size=self.batch_size, n_thread=self.num_thread)
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

            for bat_users, bat_items in data_iter.sample():
                # [(i1, i2), (i1, i2, i3)]
                split_indices = []
                for i in range(len(data_iter.batch_total_sample_sizes)):
                    if i == 0:
                        split_indices.append(data_iter.batch_total_sample_sizes[i])
                    else:
                        split_indices.append(data_iter.batch_total_sample_sizes[i] + split_indices[i-1])                        
                
                bat_neg_items = copy.deepcopy(np.split(bat_items, split_indices, 0)[:-1])
                
                for idx in range(len(split_indices)):
                    # idx = 0: bat_items_list = [i1, i2]
                    # idx = 1: bat_items_list = [i1, i2, i3]
                    bat_neg_items[idx] = np.split(bat_neg_items[idx], idx+2, 0)
                    bat_neg_items[idx].insert(len(bat_neg_items[idx]), bat_neg_items[idx][0])
                    bat_neg_items[idx].remove(bat_neg_items[idx][0])
                    bat_neg_items[idx] = np.array(bat_neg_items[idx]).ravel()
                    
                bat_neg_items = np.concatenate(bat_neg_items, axis=-1)
                
                bat_users = torch.from_numpy(bat_users).long().to(self.device)          # [batch_size]
                bat_items = torch.from_numpy(bat_items).long().to(self.device)          # [batch_size]
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)  # [batch_size]

                [user_embeddings, item_embeddings], \
                    [user_embeddings1, item_embeddings1], \
                        [user_embeddings2, item_embeddings2] = \
                            self.lightgcn(sub_graph1, sub_graph2, bat_users, bat_items, bat_neg_items)
                # ssl_logits_user [batch_size, num_users], ssl_logits_item [batch_size, num_items]
                
                batch_user_embeds = F.embedding(bat_users, user_embeddings)
                batch_pos_item_embeds= F.embedding(bat_items, item_embeddings)
                batch_neg_item_embeds = F.embedding(bat_neg_items, item_embeddings)
                
                u_splits = torch.split(batch_user_embeds, tuple(data_iter.batch_total_sample_sizes), 0)
                i_splits = torch.split(batch_pos_item_embeds, tuple(data_iter.batch_total_sample_sizes), 0)
                i_neg_splits = torch.split(batch_neg_item_embeds, tuple(data_iter.batch_total_sample_sizes), 0)
                
                pos_scores = []
                neg_scores = []
                for idx in range(len(data_iter.batch_total_sample_sizes)):
                    u_list = np.split(u_splits[idx], idx+2, 0)
                    i_list = np.split(i_splits[idx], idx+2, 0)
                    i_neg_list = np.split(i_neg_splits[idx], idx+2, 0)
                    
                    pos_scores.append(torch.mean(torch.stack([inner_product(u, i) for u, i in zip(u_list, i_list)]), axis=0))
                    neg_scores.append(torch.mean(torch.stack([inner_product(u, i) for u, i in zip(u_list, i_neg_list)]), axis=0))
                pos_scores = torch.concat(pos_scores, axis=0)
                neg_scores = torch.concat(neg_scores, axis=0)
                
                # CPR Loss
                cpr = cpr_loss(pos_scores, neg_scores, self.batch_size, self.sample_rate)
                
                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(torch.unique(bat_users)),
                    self.lightgcn.item_embeddings(torch.unique(bat_items))
                )
                
                # InfoNCE Loss
                users = torch.unique(bat_users)
                items = torch.unique(bat_items)

                user_embs1 = F.embedding(users, user_embeddings1)    # [batch_size, embedding_size]
                item_embs1 = F.embedding(items, item_embeddings1)    # [batch_size, embedding_size]
                user_embs2 = F.embedding(users, user_embeddings2)    # [batch_size, embedding_size]
                item_embs2 = F.embedding(items, item_embeddings2)    # [batch_size, embedding_size]

                user_cl_loss = InfoNCE(user_embs1, user_embs2, self.ssl_temp)
                item_cl_loss = InfoNCE(item_embs1, item_embs2, self.ssl_temp)
                
                # pos_ratings_user = inner_product(user_embs1, user_embs2)     # [batch_size]
                # pos_ratings_item = inner_product(item_embs1, item_embs2)     # [batch_size]
                # tot_ratings_user = torch.matmul(user_embs1, torch.transpose(user_embeddings2, 0, 1))    # [batch_size, num_users]
                # tot_ratings_item = torch.matmul(item_embs1, torch.transpose(item_embeddings2, 0, 1))    # [batch_size, num_items]

                # ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]
                # ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]

                # clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                # clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                
                infonce_loss = user_cl_loss + item_cl_loss
                
                loss = cpr + self.reg * reg_loss + self.ssl_reg * infonce_loss
                
                total_loss += loss
                total_sup_loss += cpr
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
        if self.best_result[0] < current_result[0]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()  # ratings

def InfoNCE(view1, view2, temperature):
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(cl_loss)
