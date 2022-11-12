# Contributors: Katie, Travis.

import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

##############################################################################

# Function definitions.


def csv_to_numpy(target_file):

    # Import csv data into a Numpy array.

    # Expected format:
    #    col0: X, col1: Y, col2: Mag, col3: Grav, col4: Grav_1vd.
    if file_exists(target_file):
        data = np.genfromtxt(target_file, delimiter=",", skip_header=1)
    else:
        data = None

    return(data)


def csv_to_pandas(target_file):

    # Import csv data into a Pandas dataframe.

    # Expected format:
    #    col0: X, col1: Y, col2: Mag, col3: Grav, col4: Grav_1vd.
    if file_exists(target_file):
        data = pd.read_csv(target_file)
    else:
        data = None

    return(data)


def file_exists(target_file):

    # Checks a file exists.

    bool_exists = os.path.isfile(target_file)
    if not(bool_exists):
        print("File not found: " + target_file)
        quit()

    return(bool_exists)


def recommend_eps(data_input, min_samples,
                  diagnostic_plots=False, plot_height=1):

    # Get a recommended epsilon to use with DBSCAN.
    # Low values require higher density to form a cluster.
    #     N.B. Too low yields no clusters; all points labelled as -1 for
    #          "noise".
    # Values too high clump all points into one cluster.
    #
    # Here a Pareto split (80%/20%) of the distances of a k-nearest neighbours
    # fit of all points is used.
    #
    # Input parameters:
    #     data_input         the data set being clustered.
    #     min_samples        integer; number of minimum neighbors being used
    #                            with DBSCAN.
    #     diagnostic_plots   boolean; a flag triggering generation of
    #                            diagnostic plots.
    #     plot_height        numeric; height in inches to use for plots.

    fit_neighbors = NearestNeighbors(n_neighbors=min_samples).fit(data_input)
    # Find the K-neighbors of each point.
    distances, indices = fit_neighbors.kneighbors(data_input)
    del(fit_neighbors)
    del(indices)

    # Here column index 0 stores the distance between the point and itself; 0.
    # Only the distance to the nearest neighbour is needed,
    # i.e. use col index 1 (the 2nd column).
    distances = distances[:, 1]

    # Sort distances from smallest to largest.
    distances = np.sort(distances, axis=0)

    # Get the distance at the Pareto boundary.
    # Split after the first 20% smallest.
    # Splitting after the first 80% produces excessive clusters.
    num_points = data_input.shape[0]
    epsilon_estimate = distances[math.floor(num_points * 0.2)]

    if diagnostic_plots:
        # Create a distance histogram.
        print("Creating a distance histogram for " + str(num_points) +
              " data points.")
        plt.axvline(epsilon_estimate)   # Overlay a vertical line for epsilon.
        seaborn.displot(distances, height=plot_height, aspect=1)
        plt.savefig("distances_histogram.pdf", dpi=300, pad_inches=0)

        # Create corresponding density estimates.
        seaborn.displot(distances, height=plot_height, aspect=1, kind="ecdf",
                        linewidth=1)
        plt.savefig("distances_ecdf.pdf", dpi=300, pad_inches=0)
        seaborn.displot(distances, height=plot_height, aspect=1, kind="kde",
                        linewidth=1)
        plt.savefig("distances_kde.pdf", dpi=300, pad_inches=0)

    return(epsilon_estimate)


def recommend_min_neighbors(data_dimensionality=1):

    # Get a recommended number of minimum neighbors to use with DBSCAN.
    # Higher values increase the density needed to form a cluster.
    # Here, the maximum is taken of:
    #    a) sklearn's default value; 5,
    #    b) twice the data dimensionality.
    #
    # Input parameters:
    #     data_dimensionality   integer; dimensionality of the data being
    #                               clustered.

    return(max(5, 2 * data_dimensionality))


def subset_data_numpy(data_input, bool_split=False):

    # Subset input Numpy data.
    # a) Drop all columns except for Mag and Grav_1vd.
    # b) Optionally, subset the data using a coorodinate-based scheme.

    if bool_split:
        # Split the data along its coordinate mid-points.
        mid_e = (max(data_input[1:, 0]) + min(data_input[1:, 0])) / 2
        mid_n = (max(data_input[1:, 1]) + min(data_input[1:, 1])) / 2

        # Split by longitude.
        data_input = data_input[data_input[:, 0] <= mid_e, :]
        #
        # Split by latitude.
        data_input = data_input[data_input[:, 1] <= mid_n, :]

    # Drop unneeded columns. Keep Mag, Grav_1vd.
    data_subset = data_input[:, [2, 4]]

    return(data_subset)


def subset_data_pandas(data_input, bool_split=False):

    # Subset input Pandas data.
    # a) Drop all columns except for Mag and Grav_1vd.
    # b) Optionally, subset the data using a coorodinate-based scheme.

    if bool_split:
        # Split the data along its coordinate mid-points.
        mid_e = (data_input[["X"]].max() + data_input[["X"]].min()) / 2
        mid_n = (data_input[["Y"]].max() + data_input[["Y"]].min()) / 2

        # Split by longitude.
        data_input = data_input[data_input["X"] <= mid_e]
        #
        # Split by latitude.
        data_input = data_input[data_input["Y"] <= mid_n]

    # Drop unneeded columns. Keep Mag, Grav_1vd.
    data_subset = data_input.loc[:, ["Mag", "Grav_1vd"]]

    return(data_subset)

##############################################################################


# Import magnetic and gravity survey data into a Numpy array.
# The data contains now duplicate rows.
# Split the data and drop unneeded columns.
print("Importing data.")
target_file = "Mag_Grav_Grav1vd.csv"
data_subset = subset_data_pandas(csv_to_pandas(target_file), bool_split=False)

# Centre and scale the data using the median and quantiles.
# N.B. RobustScaler will convert Pandas dataframes into arrays.
print("Scaling data.")
data_scaled = RobustScaler().fit(data_subset).transform(data_subset)
del(data_subset)
# Convert back to Pandas.
data_scaled = pd.DataFrame(data_scaled, columns=["Mag", "Grav_1vd"])


# Visualise the data distributions.
#
# Plot height is specified in inches.
# Try height of 20% of A4 width = 0.2 * 210 mm = 42 mm = 42/25.4 inches.
plot_height = 42/25.4
build_plots = False
#
if build_plots:
    # Mag.
    seaborn.displot(data_scaled, x="Mag", height=plot_height, aspect=1)
    plt.savefig("mag_scaled.pdf", dpi=300, pad_inches=0)
    # Grav_1vd.
    seaborn.displot(data_scaled, x="Grav_1vd", height=plot_height, aspect=1)
    plt.savefig("grav_1vd_scaled.pdf", dpi=300, pad_inches=0)
    # 2D: Mag vs Grav_1vd.
    seaborn.displot(data_scaled, x="Mag", y="Grav_1vd", height=plot_height,
                    aspect=1)
    plt.savefig("2D_mag_Grav_1vd.pdf", dpi=300, pad_inches=0)


# Convert from Pandas dataframe to numpy array for processing.
data_scaled = data_scaled.to_numpy()


# DBSCAN clustering.
print("Clustering data (" + str(data_scaled.shape[0]) + " data points).")
min_samples = recommend_min_neighbors(data_dimensionality=data_scaled.shape[1])
eps = recommend_eps(data_input=data_scaled,
                    min_samples=min_samples,
                    diagnostic_plots=build_plots,
                    plot_height=plot_height)

# Release unneeded memory.
del(plot_height)
gc.collect()
data_dbs = DBSCAN(eps=eps, min_samples=min_samples)

data_dbs.fit(data_scaled)    # Perform clustering.
dbs_labels = data_dbs.labels_ + 1


# Prepare to save results.
print("Saving results.")
output_file_prefix = target_file[0:-4] + "_dbscan_rs_e" + str(eps) \
    + "_m" + str(min_samples)
data_subset = subset_data_pandas(csv_to_pandas(target_file))   # Re-retrieve.

# Save labels.
output_file = output_file_prefix + "_labels.csv"
np.savetxt(output_file,
           np.c_[data_subset, data_scaled, dbs_labels],
           delimiter=",",
           header="Mag,Grav_1vd,Mag_scaled,Grav_1vd_scaled,Cluster_label",
           comments="")

# Save clusters.
for idx in range(1, max(dbs_labels) + 1):
    output_data = data_subset[dbs_labels == idx]
    output_file = output_file_prefix + "_cluster" + str(idx) + ".csv"
    np.savetxt(output_file,
               output_data,
               delimiter=",",
               header="Mag,Grav_1vd",
               comments="")
print("Clusters found: " + str(idx))

# Save outliers.
output_data = data_subset[dbs_labels == 0]
output_file = output_file_prefix + "_outliers.csv"
np.savetxt(output_file,
           output_data,
           delimiter=",",
           header="Mag,Grav_1vd",
           comments="")
print("Outliers found: " + str(output_data.shape[0]))

print("\nProgram complete.")
