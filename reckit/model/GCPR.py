import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from reckit.model.abstract_recommender import AbstractRecommender
from reckit import CPRSampler
from time import time
from reckit.util import l2_loss, cpr_loss, InfoNCE
from reckit.util import timer
import warnings
from reckit.util import inner_product
from reckit.util import sp_mat_to_sp_tensor
from reckit.util import get_initializer, save_pkl

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, norm_adj, n_layers=3, hyper_layers=1):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.hyper_layers = hyper_layers

        # self.user_embeddings = nn.Embedding(self.num_users, self.embedding_size)
        # self.item_embeddings = nn.Embedding(self.num_items, self.embedding_size)
        
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

    def forward(self, eps=0, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
        all_embeddings = []
        cl_embeddings_list = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * eps
            all_embeddings.append(ego_embeddings)
            cl_embeddings_list.append(ego_embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items])
        if perturbed:
            return cl_embeddings_list
        return user_all_embeddings, item_all_embeddings
    
    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")

        user_embs = F.embedding(users, self._user_embeddings_final)
        tmp_item_embeddings =self._item_embeddings_final
        ratings = torch.matmul(user_embs, tmp_item_embeddings.T)
        return ratings

    def eval(self):
        super(LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self.forward()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(init(torch.empty(self.num_users, self.embedding_size))),
            'item_emb': nn.Parameter(init(torch.empty(self.num_items, self.embedding_size))),
        })
        
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
        self.hyper_layers = config["hyper_layers"]

        # 对比学习超参数
        self.eps = config["eps"]
        self.tau = config["tau"]
        self.alpha = config["alpha"]
        self.cl_reg = config["cl_reg"]
        self.sample_ratio = config["sample_ratio"]
        self.sample_rate = config["sample_rate"]

        # 其他超参数
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)
        self.model_str = '#eps=%.2f-tau=%.2f-sample_rate=%d-sample_ratio=%d-alpha=%.2f' % (self.eps, self.tau, self.sample_rate, self.sample_ratio, self.alpha)
                         
        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = LightGCN(self.num_users, self.num_items, self.embedding_size, adj_matrix, self.n_layers, self.hyper_layers)
        self.lightgcn.reset_parameters(init_method=self.param_init)
        self.lightgcn = self.lightgcn.to(self.device)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self):
        warnings.filterwarnings('ignore')
        
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

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
            self.lightgcn.train()  # Sets the module in training mode.

            for user_idx, pos_idx in data_iter.sample():
                # [(u1, u2), (u3, u4, u5)], [(i1, i2), (i3, i4, i5)]                
                neg_idx = []
                cur_idx = 0
                for idx, size in enumerate(data_iter.batch_total_sample_sizes):
                    neg_items = pos_idx[cur_idx:cur_idx+size]
                    neg_items = np.split(neg_items, idx+2, 0)
                    neg_items.insert(len(neg_items), neg_items[0])
                    neg_items.remove(neg_items[0])
                    neg_idx.append(np.array(neg_items).ravel())
                    cur_idx += size
                neg_idx = np.concatenate(neg_idx, axis=-1)
                
                # all_user_emb, all_item_emb: 每层均值
                all_user_emb, all_item_emb = self.lightgcn()
                user_emb, pos_item_emb, neg_item_emb = all_user_emb[user_idx], all_item_emb[pos_idx], all_item_emb[neg_idx]
                
                u_splits = torch.split(user_emb, tuple(data_iter.batch_total_sample_sizes), 0)
                i_splits = torch.split(pos_item_emb, tuple(data_iter.batch_total_sample_sizes), 0)
                i_neg_splits = torch.split(neg_item_emb, tuple(data_iter.batch_total_sample_sizes), 0)
                
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
                
                # ssl loss
                embeddings_list = self.lightgcn(self.eps, True)
                u_idx = torch.unique(torch.Tensor(user_idx).type(torch.long)).cuda()
                i_idx = torch.unique(torch.Tensor(pos_idx).type(torch.long)).cuda()
                
                # 偶数层：0，2
                even_previous_embedding = embeddings_list[0]
                even_current_embedding = embeddings_list[self.hyper_layers*2]
                
                even_current_user_embeddings, even_current_item_embeddings = torch.split(even_current_embedding, [self.num_users, self.num_items])
                even_previous_user_embeddings_all, even_previous_item_embeddings_all = torch.split(even_previous_embedding, [self.num_users, self.num_items])

                even_current_user_embeddings = even_current_user_embeddings[u_idx]
                even_previous_user_embeddings = even_previous_user_embeddings_all[u_idx]
                ssl_even_user_loss = InfoNCE(even_current_user_embeddings, even_previous_user_embeddings, self.tau)
                
                even_current_item_embeddings = even_current_item_embeddings[i_idx]
                even_previous_item_embeddings = even_previous_item_embeddings_all[i_idx]
                ssl_even_item_loss = InfoNCE(even_current_item_embeddings, even_previous_item_embeddings, self.tau)
                
                # 奇数层：1，3
                odd_previous_embedding = embeddings_list[self.hyper_layers]
                odd_current_embedding = embeddings_list[self.n_layers]
                
                odd_current_user_embeddings, odd_current_item_embeddings = torch.split(odd_current_embedding, [self.num_users, self.num_items])
                odd_previous_user_embeddings_all, odd_previous_item_embeddings_all = torch.split(odd_previous_embedding, [self.num_users, self.num_items])

                odd_current_user_embeddings = odd_current_user_embeddings[u_idx]
                odd_previous_user_embeddings = odd_previous_user_embeddings_all[u_idx]
                ssl_odd_user_loss = InfoNCE(odd_current_user_embeddings, odd_previous_user_embeddings, self.tau)
                
                odd_current_item_embeddings = odd_current_item_embeddings[i_idx]
                odd_previous_item_embeddings = odd_previous_item_embeddings_all[i_idx]
                ssl_odd_item_loss = InfoNCE(odd_current_item_embeddings, odd_previous_item_embeddings, self.tau)
                
                # 合并
                ssl_loss_user = ssl_even_user_loss + self.alpha * ssl_odd_user_loss
                ssl_loss_item = ssl_even_item_loss + self.alpha * ssl_odd_item_loss
                cl_loss = self.cl_reg * (ssl_loss_user + ssl_loss_item)
                
                # Reg Loss
                reg_loss = self.reg * l2_loss(all_user_emb[u_idx], all_item_emb[i_idx])
     
                loss = cpr + reg_loss + cl_loss
                
                total_loss += loss
                total_sup_loss += cpr
                total_reg_loss += reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.6f = %.6f + %.6f + %.6f, time: %f]" % (
                epoch, 
                total_loss/ self.num_ratings,
                total_sup_loss / self.num_ratings,
                (total_loss - total_sup_loss - total_reg_loss) / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time()-training_start_time,)
            )
            if epoch % self.verbose == 0 and epoch > self.config.start_testing_epoch:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d\t%s" % (epoch, result))
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
        # self.save()
        

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
    
    def save(self):
        dir = os.path.join(os.path.realpath(r"."), 'params')
        with torch.no_grad():
            save_pkl(self.lightgcn.state_dict(), dir, 'emb')