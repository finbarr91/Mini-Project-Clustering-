# Customer Segmentation using Clustering
# This mini-project is based on this blog post by yhat. Please feel free to refer to the post for additional information, and solutions.

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Seaborn
sns.set_style("whitegrid")
sns.set_context("poster")

# Data
# The dataset contains information on marketing newsletters/e-mail campaigns
# (e-mail offers sent to customers) and transaction level data from customers.
# The transactional data shows which offer customers responded to, and what the customer ended up buying. The data is presented as an Excel workbook containing two worksheets. Each worksheet contains a different dataset.

df_offers = pd.read_excel(r"C:\Users\chukw\PycharmProjects\Mini-Project-Clustering-\WineKMC.xlsx",sheet_name=0)
df_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
print(df_offers.head(10))

# We see that the first dataset contains information about each offer such as the month it is in
# effect and several attributes about the wine that the offer refers to:
# the variety, minimum quantity, discount, country of origin and whether or not it is past peak.
# The second dataset in the second worksheet contains transactional data -- which offer each customer responded to.

df_transactions = pd.read_excel(r"C:\Users\chukw\PycharmProjects\Mini-Project-Clustering-\WineKMC.xlsx", sheet_name=1)
df_transactions.columns = ["customer_name", "offer_id"]
df_transactions['n'] = 1
print(df_transactions.head(10))

"""
Data wrangling
We're trying to learn more about how our customers behave, so we can use their behavior (whether or not they purchased something based on an offer) as a way to group similar minded customers together. We can then study those groups to look for patterns and trends which can help us formulate future offers.
The first thing we need is a way to compare customers. To do this, we're going to create a matrix that contains each customer and a 0/1 indicator for whether or not they responded to a given offer.
Checkup Exercise Set I
Exercise: Create a data frame where each row has the following columns (Use the pandas merge and pivot_table functions for this purpose):
    customer_name
    One column for each offer, with a 1 if the customer responded to the offer </ul>
    Make sure you also deal with any weird values such as `NaN`. Read the documentation to develop your solution.
"""

#your turn
df_merged=df_offers.merge(df_transactions,on='offer_id')
df_merged=df_merged.dropna()
print(df_merged.head())

a= pd.pivot_table(df_merged, index=['customer_name'],columns='offer_id', values='n', fill_value=0)
a= a.reset_index()

a_reduced=a.drop('customer_name',1)
print(a_reduced.head())

"""
K-Means Clustering

Recall that in K-Means Clustering we want to maximize the distance between centroids and minimize the distance between data points and the respective centroid for the cluster they are in. True evaluation for unsupervised learning would require labeled data; however, we can use a variety of intuitive metrics to try to pick the number of clusters K. We will introduce two methods: the Elbow method, the Silhouette method and the gap statistic.
Choosing K: The Elbow Sum-of-Squares Method

The first method looks at the sum-of-squares error in each cluster against

. We compute the distance from each data point to the center of the cluster (centroid) to which the data point was assigned.
where
is a point,
represents cluster and
is the centroid for cluster . We can plot SS vs. and choose the elbow point in the plot as the best value for
. The elbow point is the point at which the plot starts descending much more slowly.
Checkup Exercise Set II

Exercise:
    What values of 
do you believe represent better clusterings? Why?
Create a numpy matrix x_cols with only the columns representing the offers (i.e. the 0/1 colums)
Write code that applies the KMeans clustering method from scikit-learn to this matrix.
Construct a plot showing
for each and pick using this plot. For simplicity, test
Make a bar chart showing the number of points in each cluster for k-means under the best

What challenges did you experience using the Elbow method to pick?"""
# your turn
import numpy as np
b=a_reduced.as_matrix()
print(b)

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

K = range(2,10)
KM = [KMeans(n_clusters=k,random_state=0).fit(b) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(b, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/b.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(b)**2)/b.shape[0]
bss = tss-wcss

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')

plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.show()

KM = KMeans(n_clusters=3,random_state=0).fit(b)
ccc= KM.labels_
print(ccc)

bbb= {i: np.where(KM.labels_ == i)[0] for i in range(KM.n_clusters)}
print(bbb)

x_axis=[]
y_axis=[]
for i in bbb:
    #[blah,temp]=i.shape()
    j= len(bbb[i])
    x_axis.append(i)
    y_axis.append(j)

plt.bar(x_axis, y_axis, align='center', alpha=0.5)
plt.xlabel('Cluster described')
plt.ylabel('The number of points in a particular cluster')
plt.title('The number of points in each cluster for k-means under the best K')
plt.show()

"""
It will be hard to use the elbow method to detect clusters for a highly correlated dataset
Choosing K: The Silhouette Method

There exists another method that measures how well each datapoint
"fits" its assigned cluster and also how poorly it fits into other clusters. This is a different way of looking at the same objective. Denote
as the average distance from
to all other points within its own cluster . The lower the value, the better. On the other hand
is the minimum average distance from
to points in a different cluster, minimized over clusters. That is, compute separately for each cluster the average distance from
to the points within that cluster, and then take the minimum. The silhouette

is defined as

The silhouette score is computed on every datapoint in every cluster. The silhouette score ranges from -1 (a poor clustering) to +1 (a very dense clustering) with 0 denoting the situation where clusters overlap. Some criteria for the silhouette coefficient is provided in the table below.

Range 	Interpretation
0.71 - 1.0 	A strong structure has been found.
0.51 - 0.7 	A reasonable structure has been found.
0.26 - 0.5 	The structure is weak and could be artificial.
< 0.25 	No substantial structure has been found.

</pre> Source: http://www.stat.berkeley.edu/~spector/s133/Clus.html

Fortunately, scikit-learn provides a function to compute this for us (phew!) called sklearn.metrics.silhouette_score. Take a look at this article on picking

in scikit-learn, as it will help you in the next exercise set.
Checkup Exercise Set III

Exercise: Using the documentation for the `silhouette_score` function above, construct a series of silhouette plots like the ones in the article linked above.
Exercise: Compute the average silhouette score for each
and plot it. What does the plot suggest we should choose? Does it differ from what we found using the Elbow method?
"""

# Your turn.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
X=b
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

"""
Choosing
: The Gap Statistic

There is one last method worth covering for picking
, the so-called Gap statistic. The computation for the gap statistic builds on the sum-of-squares established in the Elbow method discussion, and compares it to the sum-of-squares of a "null distribution," that is, a random set of points with no clustering. The estimate for the optimal number of clusters is the value for which

falls the farthest below that of the reference distribution:
In other words a good clustering yields a much larger difference between the reference distribution and the clustered data. The reference distribution is a Monte Carlo (randomization) procedure that constructs random distributions of points within the bounding box (limits) of the original data and then applies K-means to this synthetic distribution of data points..
is just the average
over all replicates. We then compute the standard deviation
of the values of
computed from the

replicates of the reference distribution and compute
Finally, we choose such that

.
Aside: Choosing
when we Have Labels

Unsupervised learning expects that we do not have the labels. In some situations, we may wish to cluster data that is labeled. Computing the optimal number of clusters is much easier if we have access to labels. There are several methods available. We will not go into the math or details since it is rare to have access to the labels, but we provide the names and references of these measures.

    Adjusted Rand Index
    Mutual Information
    V-Measure
    Fowlkes–Mallows index

See this article for more information about these metrics.
Visualizing Clusters using PCA

How do we visualize clusters? If we only had two features, we could likely plot the data as is. But we have 100 data points each containing 32 features (dimensions). Principal Component Analysis (PCA) will help us reduce the dimensionality of our data from 32 to something lower. For a visualization on the coordinate plane, we will use 2 dimensions. In this exercise, we're going to use it to transform our multi-dimensional dataset into a 2 dimensional dataset.

This is only one use of PCA for dimension reduction. We can also use PCA when we want to perform regression but we have a set of highly correlated variables. PCA untangles these correlations into a smaller number of features/predictors all of which are orthogonal (not correlated). PCA is also used to reduce a large set of variables into a much smaller one.
Checkup Exercise Set IV

Exercise: Use PCA to plot your clusters:

    Use scikit-learn's [`PCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) function to reduce the dimensionality of your clustering data to 2 components
    Create a data frame with the following fields:
        customer name
        cluster id the customer belongs to
        the two PCA components (label them `x` and `y`) 
    Plot a scatterplot of the `x` vs `y` columns
    Color-code points differently based on cluster ID
    How do the clusters look?
    Based on what you see, what seems to be the best value for 

? Moreover, which method of choosing

    seems to have produced the optimal result visually? 

Exercise: Now look at both the original raw data about the offers and transactions and look at the fitted clusters. Tell a story about the clusters in context of the original data. For example, do the clusters correspond to wine variants or something else interesting?
"""
names=[]
for i in bbb:
    for j in bbb[i]:
        names.append([df_transactions.iloc[j,0], i])
names

ff= pd.DataFrame(names, columns=['customer_id', 'cluster_id'])

#your turn

from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca.fit(a_reduced)
a_reduced = pca.transform(a_reduced)
print(a_reduced)

dd= pd.DataFrame(a_reduced, columns=['x', 'y'])

ff[2]=dd['x']
ff[3]=dd['y']
ff.columns= ['customer_id','cluster_id','x','y']
print(ff.head())

labels = np.array(KM.labels_)
labels = labels.tolist()
LABEL_COLOR_MAP = ['r', 'k', 'b']
label_color = []
for l in labels:
    label_color.append(LABEL_COLOR_MAP[l])

plt.scatter(ff['x'], ff['y'], c=label_color)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""


3 clusters emerges as the optimal and this is made visually more obvious by using The Elbow Sum-of-Squares Method to plot our clusters.

What we've done is we've taken those columns of 0/1 indicator variables, and we've transformed them into a 2-D dataset. We took one column and arbitrarily called it x and then called the other y. Now we can throw each point into a scatterplot. We color coded each point based on it's cluster so it's easier to see them.
Exercise Set V

As we saw earlier, PCA has a lot of other uses. Since we wanted to visualize our data in 2 dimensions, restricted the number of dimensions to 2 in PCA. But what is the true optimal number of dimensions?

Exercise: Using a new PCA object shown in the next cell, plot the `explained_variance_` field and look for the elbow point, the point where the curve's rate of descent seems to slow sharply. This value is one possible value for the optimal number of dimensions. What is it?
"""

#your turn
# Initialize a new PCA model with a default number of components.
import sklearn.decomposition
pca = sklearn.decomposition.PCA()
pca.fit(b)

# Do the rest on your own :)
dimensionsPossible = range(1,len(pca.explained_variance_)+1)
plt.plot(dimensionsPossible, pca.explained_variance_)
plt.xlabel('Number of Dimensions Possible')
plt.ylabel('Explained Variance')
plt.show()

"""
Other Clustering Algorithms

k-means is only one of a ton of clustering algorithms. Below is a brief description of several clustering algorithms, and the table provides references to the other clustering algorithms in scikit-learn.

    Affinity Propagation does not require the number of clusters 

    to be known in advance! AP uses a "message passing" paradigm to cluster points based on their similarity.

    Spectral Clustering uses the eigenvalues of a similarity matrix to reduce the dimensionality of the data before clustering in a lower dimensional space. This is tangentially similar to what we did to visualize k-means clusters using PCA. The number of clusters must be known a priori.

    Ward's Method applies to hierarchical clustering. Hierarchical clustering algorithms take a set of data and successively divide the observations into more and more clusters at each layer of the hierarchy. Ward's method is used to determine when two clusters in the hierarchy should be combined into one. It is basically an extension of hierarchical clustering. Hierarchical clustering is divisive, that is, all observations are part of the same cluster at first, and at each successive iteration, the clusters are made smaller and smaller. With hierarchical clustering, a hierarchy is constructed, and there is not really the concept of "number of clusters." The number of clusters simply determines how low or how high in the hierarchy we reference and can be determined empirically or by looking at the dendogram.

    Agglomerative Clustering is similar to hierarchical clustering but but is not divisive, it is agglomerative. That is, every observation is placed into its own cluster and at each iteration or level or the hierarchy, observations are merged into fewer and fewer clusters until convergence. Similar to hierarchical clustering, the constructed hierarchy contains all possible numbers of clusters and it is up to the analyst to pick the number by reviewing statistics or the dendogram.

    DBSCAN is based on point density rather than distance. It groups together points with many nearby neighbors. DBSCAN is one of the most cited algorithms in the literature. It does not require knowing the number of clusters a priori, but does require specifying the neighborhood size.

Clustering Algorithms in Scikit-learn


Exercise Set VI

Exercise: Try clustering using the following algorithms.

    Affinity propagation

    Spectral clustering

    Agglomerative clustering

    DBSCAN </ol>

    How do their results compare? Which performs the best? Tell a story why you think it performs the best.
"""
# Your turn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(b)

# Density-Based Spatial Clustering of Applications with Noise
db = DBSCAN(eps=0.2, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
"""
We get 3 clusters again however, DBSCAN is dependent on the hyperparameter selection of 'eps' and 'min_samples'. 
Using The Elbow Sum-of-Squares Method still seems like a better option since it is easy to setup and hyperparameter selection is more direct.
"""