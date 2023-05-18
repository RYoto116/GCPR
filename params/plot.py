import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import manifold
from sklearn.preprocessing import normalize
import seaborn as sns
from math import pi

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

with open("./params/emb.pkl", "rb") as f:
    embedding_dict = pickle.load(f)

user_emb = embedding_dict['embedding_dict.user_emb']
item_emb = embedding_dict['embedding_dict.item_emb']
name = 'CrossGCL'

udx = np.random.choice(len(user_emb), 2000)
selected_user_emb = user_emb[udx]
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
user_emb_2d = tsne.fit_transform(selected_user_emb.cpu())
user_emb_2d = normalize(user_emb_2d, axis=1, norm='l2')
data = {}
data[name]=user_emb_2d

f, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
kwargs = {'levels': np.arange(0, 5.5, 0.5)}
# for i,name in enumerate(models):
sns.kdeplot(data=data[name], bw=0.05, shade=True, cmap="GnBu", legend=True, ax=axs[0], **kwargs)
axs[0].set_title(name, fontsize=9, fontweight="bold")
x = [p[0] for p in data[name]]
y = [p[1] for p in data[name]]
angles = np.arctan2(y, x)
sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axs[1], color='green')

# for ax in axs:
axs[0].tick_params(axis='x', labelsize=8)
axs[0].tick_params(axis='y', labelsize=8)
axs[0].patch.set_facecolor('white')
axs[0].collections[0].set_alpha(0)
axs[0].set_xlim(-1.2, 1.2)
axs[0].set_ylim(-1.2, 1.2)
axs[0].set_xlabel('Features', fontsize=9)
axs[0].set_ylabel('Features', fontsize=9)

# for ax in axs[1]:
axs[1].tick_params(axis='x', labelsize=8)
axs[1].tick_params(axis='y', labelsize=8)
axs[1].set_xlabel('Angles', fontsize=9)
axs[1].set_ylim(0, 0.5)
axs[1].set_xlim(-pi, pi)
axs[1].set_ylabel('Density', fontsize=9)

plt.show()
plt.savefig(name + '.jpg')
