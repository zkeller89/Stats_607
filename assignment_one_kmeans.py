# Assignment 1, Part 1: K-means
# Version 0.1
#
# Thanks to the following students for reporting bugs:

import random


# TASK 1.1
# Complete the function definition below so that the functions
# returns a tuple containing two lists: the first list should be
# a list of instance (themselves represented as lists) and the
# second list should be a list of labels.
def read_data(filename):
    """ Reads instances and labels from a file. """

    f = open(filename, 'r')
    instances = []
    labels = []

    for line in f:

        # read both feature values and label
        instance_and_label = [float(x) for x in line.split()]

        # TASK 1.1.1
        # Remove label (last item) from instance_and_label and append it
        # to labels
        labels.append(instance_and_label.pop())

        # TASK 1.1.2
        # Append the instance to instances
        instances.append(instance_and_label)

    return instances, labels


# TASK 1.2
# Complete the function definition below so that the function returns the
# number of unique elements in the list labels
def num_unique_labels(labels):
    """ Return number of unique elements in the list labels. """

    return len(set(labels))


# TASK 1.3
# Complete the function definition below so that K centers get chosen from
# the given instances using the initialization used by the kmeans++ algorithm
# as described, for example, in:
# http://en.wikipedia.org/wiki/K-means%2B%2B#Initialization_algorithm
# (Steps 1 through 4)
def kmeans_plus_plus(instances, K):
    """ Choose K centers from instances using the kmeans++ initialization. """

    centers = []
    insts = instances[:]

    # select random center - step 1
    cntr = random.choice(insts)
    centers.append(cntr)
    insts.remove(cntr)

    # loop k-1 times to add k-1 centers (note we selected 1st at random)
    for n in xrange(K-1):
        probs = []
        # assign probabilities as min euclidean dist from nearest center
        for inst in insts:
            distances = [euclidean_squared(inst, x) for x in centers]
            probs.append(min(distances))

        # select new center based on weighted probabilities
        x = random.uniform(0, sum(probs))
        cum_prob = 0.0

        for item, prob in zip(insts, probs):
            cum_prob += prob
            if x < cum_prob:
                centers.append(item)
                insts.remove(item)
                break

    return centers


def euclidean_squared(p1, p2):
    """ Return squared Euclidean distance between two points p1 and p2. """

    return sum([abs(x-y)**2 for (x, y) in zip(p1, p2)])


# TASK 1.4
# Complete the function definition below to return a list cluster_ids
# such that cluster_idx[i] is the index of the center closest to instances[i].
def assign_cluster_ids(instances, centers):
    """ Assigns each instance the id of the center closest to it. """

    n = len(instances)
    cluster_ids = n*[0]  # create list of zeros

    for i in range(n):

        # TASK 1.4.1
        # Compute distances of instances[i] to each of the centers using a list
        # comprehension. Make use of the euclidean_squared function defined
        # above.
        distances = [euclidean_squared(instances[i], x) for x in centers]

        # Find the minimum distance.
        min_distance = min(distances)

        # TASK 1.4.2
        # Set the cluster id to be the index at which min_distance
        # is found in the list distances.
        cluster_ids[i] = distances.index(min_distance)

    return cluster_ids


# TASK 1.5
# Complete the function definition below to recompute the centers given cluster
# ids of instances.
def recompute_centers(instances, cluster_ids, centers):
    """ Compute centers (means) given cluster ids. """

    K = len(centers)
    n = len(cluster_ids)

    for i in range(K):

        # TASK 1.5.1
        # Find indices of of those instances whose cluster id is i.
        # Use a single list comprehension.
        one_cluster = [instances.index(x) for x, y in
                       zip(instances, cluster_ids) if y == i]
        cluster_size = len(one_cluster)
        if cluster_size == 0:  # empty cluster
            raise Exception("kmeans: empty cluster created.")

        # TASK 1.5.2
        # Suppose one_cluster is [i1, i2, i3, ... ]
        # Compute the mean of the points instances[i1], instances[i2], ...
        # using a call to reduce().
        # Supply the right 1st arg: a lambda function (this should take two
        # points [represented as lists] as arguments and return their sum) and
        # the right 2nd arg: a list (computed using a list comprehension)
        sum_cluster = reduce(lambda x, y:
                             [a + b for a, b in zip(x, y)],
                             [instances[ind] for ind in one_cluster])
        centers[i] = [x/cluster_size for x in sum_cluster]


def cluster_using_kmeans(instances, K, init='random'):
    """ Cluster instances using the K-means algorithm.

    The init argument controls the initial clustering.
    """

    err_message = 'Expected init to be "random" or "kmeans++", got %s'
    if init != 'random' and init != 'kmeans++':
        raise Exception(err_message % init)

    if init == 'random':
        # Choose initial centers at random from the given instances
        centers = random.sample(instances, K)
    else:
        # Assign clusters using the kmeans++ enhancement.
        centers = kmeans_plus_plus(instances, K)

    # create initial cluster ids
    cluster_ids = assign_cluster_ids(instances, centers)

    converged = False
    while not converged:

        # recompute centers; note function returns None, modifies centers
        # directly
        recompute_centers(instances, cluster_ids, centers)

        # re-assign cluster ids
        new_cluster_ids = assign_cluster_ids(instances, centers)

        if new_cluster_ids == cluster_ids:  # no change in clustering
            converged = True
        else:
            cluster_ids = new_cluster_ids

    return cluster_ids, centers


def main():

    data_file = 'seeds_dataset.txt'
    instances, labels = read_data(data_file)
    print 'Read %d instances and %d labels from file %s.' \
        % (len(instances), len(labels), data_file)

    if len(instances) != len(labels):
        raise Exception('Expected equal number of instances and labels.')
    else:
        n = len(instances)

    # Find number of clusters by finding out how many unique elements are there
    # in labels.
    K = num_unique_labels(labels)
    print 'Found %d unique labels.' % K

    # Run k-means clustering to cluster the instances.
    cluster_ids, centers = cluster_using_kmeans(instances, K)

    # Print the provided labels and the found clustering
    print "Done with kmeans.\nPrinting instance_id, label, cluster_id."
    for i in range(n):
        print '%3d %2d %2d' % (i, labels[i], cluster_ids[i])

    # Now run k-means using kmeans++ initialization
    cluster_ids, centers = cluster_using_kmeans(instances, K, 'kmeans++')

    # Print the provided labels and the found clustering
    print "Done with kmeans++.\nPrinting instance_id, label, cluster_id."
    for i in range(n):
        print '%3d %2d %2d' % (i, labels[i], cluster_ids[i])


if __name__ == '__main__':
    main()
