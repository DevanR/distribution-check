# Distibition-Check Version 0.2

This is a simple python script that checks a set of samples against more than
80 different distributions. See also [www.aizac.info](http://www.aizac.info/simple-check-of-a-sample-against-80-distributions) for more information.

## Requirements

* python 2.x or 3.x
* scipy
* joblib
* multiprocessing
* optparse
* and matplotlib

## Usage

Getting help:
```
$ python distribution_check.py --help
Options:
  -h, --help            show this help message and exit
  -f FILE, --file=FILE  file with measurement data
  -v, --verbose         print all results immediately (default=False)
  -t TOP, --top=TOP     define amount of printed results (default=10)
  -p, --plot            plot the best result with matplotlib (default=False)
  -i ITERATIVE, --iterative=ITERATIVE
                        define number of iterative checks (default=1)
  -e EXCLUDE, --exclude=EXCLUDE
                        amount (in per cent) of exluded samples for each
                        iteration (default=10.0%)
  -n PROCESSES, --processes=PROCESSES
                        number of process used in parallel (default=-1...all)
  -d, --densities
  -g, --generate        generate an example file
```

An example file can be generated with the parameter ```--generate```, it can
be tested afterwards with (```-n 4``` is only used to use 4 processes):

```
$ python distribution_check.py --file example-halflogistic.dat -n 4
-------------------------------------------------------------------
Top 10 after 1 iteration(s)
-------------------------------------------------------------------
1    genexpon         p:  0.998344974725      D:  0.0180427379903
2    frechet_r        p:  0.977129253832      D:  0.0222550798576
3    weibull_min      p:  0.977129253832      D:  0.0222550798576
4    beta             p:  0.965994889690      D:  0.0232109175700
5    exponweib        p:  0.931089942195      D:  0.0253029293963
6    gengamma         p:  0.922662163974      D:  0.0257070026670
7    powerlognorm     p:  0.904071427672      D:  0.0265218917625
8    f                p:  0.890318294041      D:  0.0270739589735
9    betaprime        p:  0.890296634662      D:  0.0270748009279
10   halflogistic     p:  0.890060950383      D:  0.0270839574382
```

The halflogistic is only on the 10th place. Since the applied "Kolmogorov-Smirnof one sided test" is applied, which itself uses random
generated values, the result can be further relativized by a couple of
iterations.

```
$ python distribution_check.py --file example-halflogistic.dat -n 4 -i 50
-------------------------------------------------------------------
Top 10 after 50 iteration(s)
-------------------------------------------------------------------
1    genexpon         p: 0.98646..   D: 0.02145..   var(p): 0.03843..   var(D):  0.00006010..
2    halflogistic     p: 0.93195..   D: 0.02549..   var(p): 0.01172..   var(D):  0.00001921..
3    genhalflogistic  p: 0.90430..   D: 0.02667..   var(p): 0.13291..   var(D):  0.00292669..
4    frechet_r        p: 0.88877..   D: 0.02728..   var(p): 0.12868..   var(D):  0.00049114..
5    weibull_min      p: 0.88877..   D: 0.02728..   var(p): 0.12868..   var(D):  0.00049114..
6    beta             p: 0.86228..   D: 0.02811..   var(p): 0.05033..   var(D):  0.00027424..
7    ncx2             p: 0.78582..   D: 0.03078..   var(p): 0.08813..   var(D):  0.00020365..
8    exponweib        p: 0.75905..   D: 0.03172..   var(p): 0.11971..   var(D):  0.00036444..
9    gengamma         p: 0.74758..   D: 0.03179..   var(p): 0.11281..   var(D):  0.00036131..
10   mielke           p: 0.72108..   D: 0.03262..   var(p): 0.02134..   var(D):  0.00001869..
```
The last two values represent the variance of the p and D value, their
thightness can also be used as an indicator vor the averaged values (of p and
D), the smaller the variance the more accurate should be the result.

Check out the other paramters ...
