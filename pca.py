import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.interpolate import griddata


def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    plt.grid()
    plt.xlim(xs.min()/2.5, xs.max()/2.5)
    plt.ylim(ys.min()/2, ys.max()/2)
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c='y')
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                     "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                     labels[i], color='g', ha='center', va='top')


pca = PCA()

df = pd.read_excel('DA-E-191029.xlsx', sheet_name=1)
df = df.apply(scipy.stats.zscore, axis=0, ddof=1)
label = ['Grades at school', 'Aptitude test', 'Interview test']
X = df[['在学時成績', '適性検査', '面接試験']]
pca.fit(X)
score = pca.fit_transform(X)
print(score[:, 0:2])

plt.plot(np.hstack([0, pca.explained_variance_ratio_]
                   ).cumsum(), 'o-')  # 0からプロット
plt.xticks(range(4))
plt.xlabel('components')
plt.ylabel('explained variance ratio')
plt.grid()
plt.figure()


plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)],
        pca.explained_variance_ratio_)
plt.title('Contribution Rate')
plt.figure()
biplot(score[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=label)

avg = np.array([np.mean(X['在学時成績']), np.mean(X['適性検査']), np.mean(X['面接試験'])])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel('Grades at school')
ax.set_ylabel('Aptitude test')
ax.set_zlabel('Interview test')

distance = np.sqrt(((X.max(axis=0) - X.min(axis=0)) ** 2).sum())
right = distance / 2
left = -right

xs0 = np.array([pca.components_[0][0] * left,
                pca.components_[0][0] * right]) + avg[0]
ys0 = np.array([pca.components_[0][1] * left,
                pca.components_[0][1] * right]) + avg[1]
zs0 = np.array([pca.components_[0][2] * left,
                pca.components_[0][2] * right]) + avg[2]

xs1 = np.array([pca.components_[1][0] * left,
                pca.components_[1][0] * right]) + avg[0]
ys1 = np.array([pca.components_[1][1] * left,
                pca.components_[1][1] * right]) + avg[1]
zs1 = np.array([pca.components_[1][2] * left,
                pca.components_[1][2] * right]) + avg[2]

xs2 = np.array([pca.components_[2][0] * left,
                pca.components_[2][0] * right]) + avg[0]
ys2 = np.array([pca.components_[2][1] * left,
                pca.components_[2][1] * right]) + avg[1]
zs2 = np.array([pca.components_[2][2] * left,
                pca.components_[2][2] * right]) + avg[2]

ax.plot(xs0, ys0, zs0, 'r', label='PC1')
ax.plot(xs1, ys1, zs1, 'g', label='PC2')
ax.plot(xs2, ys2, zs2, 'b', label='PC3')
ax.scatter(score[:, 0], score[:, 1], score[:, 2])
ax.legend()
plt.show()
