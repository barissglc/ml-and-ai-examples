# scikit-learn kütüphanesi ile Rastgele Veri Üretelim
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np


X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)

print(X)
print("---")
print(y)
print("---")
print(X.shape)
print("---")
print(y.shape)
print("---")

# Rastgele Verimize Bakalım
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
plt.show()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0).fit(X, y)

def visualize_classifier(model, X, y):
    fig= plt.figure(figsize = (16,8))
    
    ax = plt.gca()
    

    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='rainbow',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', zorder=1)

    ax.set(xlim=xlim, ylim=ylim)

    plt.show()

visualize_classifier(tree, X, y)



tree = DecisionTreeClassifier(random_state=99).fit(X, y)

visualize_classifier(tree, X, y)

import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X,y)

# Plot
visualize_classifier(forest, X, y)