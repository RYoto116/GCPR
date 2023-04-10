#ifndef SAMPLE_H
#define SAMPLE_H
#include <vector>
#include <unordered_set>
#include <iostream>
#include "thread_pool.h"

using namespace std;

int CPRSample1Batch(int *interact_idx,
                    const vector<unordered_set<int>> &train,
                    int *u_interacts,
                    int *i_interacts,
                    int *users,
                    int *items,
                    int k, // k-interaction samples
                    int batch_choice_size,
                    int batch_sample_size)
{
    int len = 0;
    int curr_u, next_i;
    for (int i = 0; i < batch_choice_size; i++)
    {
        bool flag = true;
        for (int j = 0; j < k; j++)
        {
            curr_u = u_interacts[interact_idx[i + batch_choice_size * j]];
            next_i = i_interacts[interact_idx[i + batch_choice_size * ((j + 1) % k)]];
            if (train[curr_u].find(next_i) != train[curr_u].end())
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            for (int j = 0; j < k; j++)
            {
                users[len + batch_sample_size * j] = u_interacts[interact_idx[i + batch_choice_size * j]];
                items[len + batch_sample_size * j] = i_interacts[interact_idx[i + batch_choice_size * j]];
            }
            len++;
            if (len == batch_sample_size)
                break;
        }
    }
    if (len < batch_sample_size)
        return -1;
    return 0;
}

class CppCPRSampler
{
public:
    CppCPRSampler(){};
    CppCPRSampler(const vector<unordered_set<int>> &train,
                  int *u_interacts,
                  int *i_interacts,
                  int *users,
                  int *items,
                  int n_step,
                  int *batch_sample_sizes,
                  int sizes_len,
                  int n_thread)
        : train(train),
          u_interacts(u_interacts),
          i_interacts(i_interacts),
          users(users),
          items(items),
          n_step(n_step),
          batch_sample_sizes(batch_sample_sizes),
          sizes_len(sizes_len),
          n_thread(n_thread) {}

    int Sample(int *interact_idx, int interact_idx_len, int *batch_choice_sizes)
    {
        ThreadPool pool(n_thread);
        vector<future<int>> results;
        int *interact_pt = interact_idx;
        int *u_pt = users;
        int *i_pt = items;

        for (int i = 0; i < n_step; i++)
        {
            for (int j = 0; j < sizes_len; j++)
            {
                results.emplace_back(
                    pool.enqueue(CPRSample1Batch,
                                 interact_pt,
                                 cref(train),
                                 u_interacts,
                                 i_interacts,
                                 u_pt,
                                 i_pt,
                                 j + 2,
                                 batch_choice_sizes[j],
                                 batch_sample_sizes[j]));
                interact_pt += (j + 2) * batch_choice_sizes[j];
                u_pt += (j + 2) * batch_sample_sizes[j];
                i_pt += (j + 2) * batch_sample_sizes[j];
            }
        }

        int status = 0;
        for (auto &&result : results)
        {
            if (result.get() == -1)
            {
                status = -1;
            }
        }
        return status;
    }

    vector<unordered_set<int>> train;
    int *u_interacts;
    int *i_interacts;
    int *users;
    int *items;
    int n_step;
    int *batch_sample_sizes;
    int sizes_len;
    int n_thread;
};
