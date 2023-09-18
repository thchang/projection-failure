import csv
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy import linalg

# --- Define utility functions --- #

def group_neighbors(df, tol, p=np.inf, show=False):
    """ Utility function to group points whose distance is less than tol.

    Copied from:
        https://www.tutorialguruji.com/python/finding-duplicates-with-tolerance-and-assign-to-a-set-in-pandas/

    Args:
        df (pandas.DataFrame): a DataFrame object containing the points to
            group.

        tol (pandas.Series): a Series object specifying the tolerance for
            grouping accross all dimensions.

        p (int, optional): the p-norm to use for groupings (after
            rescaling by the tolerance). Defaults to the infinity-norm.

        show (bool, optional): display the clustered network computed
            via networkx.

    Returns:
        pandas.DataFrame: the grouped DataFrame object.

    """

    r = np.linalg.norm(np.ones(len(tol)), p)
    print("==> Making kd tree")
    kd = KDTree(df[tol.index] / tol)
    print("==> Done making kd tree")
    g = nx.Graph([
        (i, j)
        for i, neighbors in enumerate(kd.query_ball_tree(kd, r=r, p=p))
        for j in neighbors
    ])
    print("==> Done making neighbor graph")
    if show:
        nx.draw_networkx(g)
    ix, id_ = np.array([
        (j, i)
        for i, s in enumerate(nx.connected_components(g))
        for j in s
    ]).T
    id_[ix] = id_.copy()
    return df.assign(set_id=id_)


def __rescale_x(train_x, test_x):
    """ Rescale training and testing points to improve problem conditioning.

    After re-scaling, drops redundant columns and applies PCA to remove
    subspaces.

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

    Returns:
        rescale_train (numpy.ndarray), rescale_test (numpy.ndarray):
        The two rescaled datasets, rescaled dimension-wise to fill
        the cube [-1, 1]

    """

    # Shift and scale to the unit sphere
    __shift__ = np.zeros(train_x.shape[1])
    __scale__ = np.ones(train_x.shape[1])
    __shift__[:] = np.mean(train_x, axis=0)
    __scale__[:] = np.max(np.array([np.abs(xi - __shift__[:])
                                    for xi in train_x]), axis=0)
    # Rescale the training/testing data
    rescaled_train = np.array([(xi - __shift__) / __scale__ for xi in train_x])
    rescaled_test = np.array([(xi - __shift__) / __scale__ for xi in test_x])
    # Drop columns with zero variation
    j = 0
    for i, ui in enumerate(__scale__):
        if ui < 1.0e-10:
            rescaled_train = np.delete(rescaled_train, i-j, 1)
            rescaled_test = np.delete(rescaled_test, i-j, 1)
            j += 1
    # Apply PCA to remaining columns
    print("Starting PCA")
    red_mat = np.dot(rescaled_train.T, rescaled_train)
    print(f" ==> Size: {red_mat.shape}")
    sig, v = linalg.eigh(red_mat)
    d_reduced = 0
    for si in sig:
        print(si / sig[-1])
        if si / sig[-1] > 1.0e-4:
            d_reduced += 1
    print(f" ==> reduced-dim: {d_reduced}")
    rescaled_train = np.dot(rescaled_train, v[:, -d_reduced:])
    rescaled_test = np.dot(rescaled_test, v[:, -d_reduced:])
    print("Done.")
    return rescaled_train, rescaled_test


#=============================================================================#
# Load the KDDCUP data and de-duplicate rows that are the same up to tolerance.
#=============================================================================#

remove_dupes = True

# Load the KDDCUP training data
d = 0; n = 0; m = 0
train = []
test = []
with open("KDDCUP99_raw.dat", "r") as fp:
    csv_reader = csv.reader(fp)
    for i, rowi in enumerate(csv_reader):
        if i == 0:
            name = rowi[0]
        elif i == 1:
            d = int(rowi[0].strip())
            n = int(rowi[1].strip())
            m = int(rowi[2].strip())
        else:
            xi = [float(colj) for colj in rowi]
            if i - 2 < n:
                train.append(xi)
            else:
                test.append(xi)
train = np.asarray(train)
test = np.asarray(test)
train, test = __rescale_x(train,test)

# Get metadata info
ndims = train.shape[1]
colnames = [f"x{i+1}" for i in range(ndims)]

# Convert to pandas dataframe
full_dataset = pd.DataFrame(data=train, columns=colnames)
inputdf = pd.DataFrame(data=train, columns=colnames)

if remove_dupes:

    # for uniform_tol in [1e-7,1e-6,1e-5,1e-4,1e-3]:
    uniform_tol = 1e-4
    array_tol = uniform_tol * np.ones(len(full_dataset.columns))
    tol = pd.Series(array_tol, index=inputdf.columns)

    gndf = group_neighbors(inputdf, tol, p=2)

    print("==> gndf = ")
    print(gndf)
    print("==> gndf.dup(set id) = ")
    print(gndf.duplicated(subset='set_id'))
    print("==> sum(gndf.dup(set id)) = ")
    print(sum(gndf.duplicated(subset='set_id')))
    print("==> fd[gndf.dup(set id)] = ")
    print(full_dataset[gndf.duplicated(subset='set_id') == False])

    full_dataset['set_id'] = gndf['set_id']
    #np.save('hydra_with_dupe_ids.npy', full_dataset.to_numpy())
    #print("==> SAVED full_dataset with dupes ID'd to hydra_with_dupe_ids.npy") 

    #print("==> adding dupes column and get")
    #print(full_dataset)
    #full_dataset = full_dataset[full_dataset['dupes'] == False].reset_index(drop=True) 
    #print("==> remove  dupes and get")
    #print(full_dataset)

    ## THIS NEXT LINE IS CRUCIAL: 
    # it groups by the neighborhood clusters, 
    # then creates a new point with the mean of that cluster
    # then replaces "full_dataset" with the re-assigned data
    full_dataset = full_dataset.groupby(['set_id']).mean().reset_index(drop=True)
    print("==> For tolerance = ", array_tol, " there are ", len(full_dataset), " distinct inputs.")

    # Save the filtered data
    with open("KDDCUP99_clean.dat", "w") as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow([name])
        csv_writer.writerow([len(full_dataset.columns),len(full_dataset),m])
        for i, rowi in full_dataset.iterrows():
            csv_writer.writerow([rowi[f"x{j}"] for j in range(1,len(full_dataset.columns)+1)])
        for row in test:
            csv_writer.writerow(row)

