"""
Checks a sample against 80 distributions by applying the Kolmogorov-Smirnov test.
"""

from __future__ import print_function

import math
import multiprocessing
import operator
import random
import warnings
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from joblib import Parallel, delayed

warnings.simplefilter('ignore')


def get_all_cdfs():
    """
    This function returns a dict of all available
    continuos distribution functions in the scipy.stats
    library.
    """
    dist_continuous = [
        d for d in dir(scipy.stats)
        if isinstance(getattr(scipy.stats, d), scipy.stats.rv_continuous)
    ]

    print("{} distributions found.".format(len(dist_continuous)))

    cdfs = {}
    for d in dist_continuous:
        # Currently not implemented for this distribution
        if d not in ["levy_stable"]:
            cdfs[d] = {}
            cdfs[d]["p"] = []
            cdfs[d]["D"] = []

    return cdfs


def read_file(filename):
    """
    This function reads in the sample data from a text file.
    """
    if not options.filename == "":
        print("Reading in data from %s ... " % options.filename, end="")
        f = open(options.filename)
        data = [float(value) for value in f.readlines()]
        f.close()
        print("Done!")

    return data


def read_pickle(filename):
    """
    This function reads in the sample data from a python pickle.
    """
    if not options.filename == "":
        import pandas as pd
        print("Reading in data from %s ... " % options.filename, end="")
        pickle = pd.read_pickle(options.filename)
        data = pickle['OUT_USAGE_GB']

        print("Done!")

    return data


def check(data, fct, verbose=False):
    """
    Evaluates fit of data against every know distribution by applying the
    Kolmogorov-Smirnof two-sided test.
    """
    parameters = eval("scipy.stats." + fct + ".fit(data)")
    D, p = scipy.stats.kstest(data, fct, args=parameters)

    if math.isnan(p):
        p = 0
    if math.isnan(D):
        D = 0

    if verbose:
        print(fct.ljust(16) + "p: " + str(p).ljust(25) + "D: " + str(D))

    return (fct, p, D)


def plot(fcts, data, filename):
    """
    Plots data against all known distributions.
    """
    plt.hist(data, normed=True, bins=max(10, len(data) / 10))

    for fct in fcts:
        params = eval("scipy.stats." + fct + ".fit(data)")
        f = eval("scipy.stats." + fct + ".freeze" + str(params))
        x = np.linspace(f.ppf(0.001), f.ppf(0.999), 500)
        plt.plot(x, f.pdf(x), lw=3, label=fct)
    plt.legend(loc='best', frameon=False)
    plt.title("Top " + str(len(fcts)) + " Results")
    # plt.show()
    plt.savefig(filename + '.png', bbox_inches='tight')


def plotDensities(best):
    """
    Plots data densities against all known distribution densities.
    """
    plt.ion()
    plt.clf()

    for i in range(len(best) - 1, -1, -1):
        fct, values = best[i]
        plt.hist(
            values["p"],
            normed=True,
            bins=max(10, len(values["p"]) / 10),
            label=str(i + 1) + ". " + fct,
            alpha=0.5)
    plt.legend(loc='best', frameon=False)
    plt.title("Top Results")
    plt.show()
    plt.draw()


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        default="",
        type="string",
        help="file with measurement data",
        metavar="FILE")
    parser.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="print all results immediately (default=False)")
    parser.add_option(
        "-t",
        "--top",
        dest="top",
        default=10,
        type="int",
        help="define amount of printed results (default=10)")
    parser.add_option(
        "-p",
        "--plot",
        dest="plot",
        default=False,
        action="store_true",
        help="plot the best result with matplotlib (default=False)")
    parser.add_option(
        "-i",
        "--iterative",
        dest="iterative",
        default=1,
        type="int",
        help="define number of iterative checks (default=1)")
    parser.add_option(
        "-e",
        "--exclude",
        dest="exclude",
        default=10.0,
        type="float",
        help="amount (in per cent) of exluded samples for each iteration (default=10.0%)"
    )
    parser.add_option(
        "-n",
        "--processes",
        dest="processes",
        default=-1,
        type="int",
        help="number of process used in parallel (default=-1...all)")
    parser.add_option(
        "-d",
        "--densities",
        dest="densities",
        default=False,
        action="store_true",
        help="")
    parser.add_option(
        "-g",
        "--generate",
        dest="generate",
        default=False,
        action="store_true",
        help="generate an example file")

    (options, args) = parser.parse_args()

    if options.generate:
        print("generating random data 'example-halflogistic.dat' ... ", end="")
        f = open("example-halflogistic.dat", "w")
        f.writelines(
            [str(s) + "\n" for s in scipy.stats.halflogistic().rvs(500)])
        f.close()
        print("done")
        quit()

    # read data from file or generate
    #DATA = read_file(options.filename)
    DATA = read_pickle(options.filename)

    CDFS = get_all_cdfs()

    for i in range(options.iterative):

        if options.iterative == 1:
            data = DATA
        else:
            data = [
                value for value in DATA
                if random.random() >= options.exclude / 100
            ]

        results = Parallel(n_jobs=options.processes)(
            delayed(check)(data, fct, options.verbose) for fct in CDFS.keys())

        for res in results:
            key, p, D = res
            CDFS[key]["p"].append(p)
            CDFS[key]["D"].append(D)

        print(
            "-------------------------------------------------------------------"
        )
        print("Top %d after %d iteration(s)" % (
            options.top,
            i + 1, ))
        print(
            "-------------------------------------------------------------------"
        )
        best = sorted(
            CDFS.items(),
            key=lambda elem: scipy.median(elem[1]["p"]),
            reverse=True)

        for t in range(options.top):
            fct, values = best[t]
            print(
                str(t + 1).ljust(4),
                fct.ljust(16),
                "\tp: ",
                scipy.median(values["p"]),
                "\tD: ",
                scipy.median(values["D"]),
                end="")
            if len(values["p"]) > 1:
                print(
                    "\tvar(p): ",
                    scipy.var(values["p"]),
                    "\tvar(D): ",
                    scipy.var(values["D"]),
                    end="")
            print()

        if options.densities:
            plotDensities(best[:t + 1])

    if options.plot:
        # get only the names ...
        plot([b[0] for b in best[:options.top]], DATA, options.filename)
