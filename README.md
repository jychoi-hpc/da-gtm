DA-GTM
======

Solving GTM with Deterministic Annealing

Introduction
------------

Generative Topographic Mapping (GTM) is an algorithm for data
visualization through dimension reduction. Unlike PCA, which is a
traditional visualization method based on linear algebra, GTM seeks a
non-linear mapping. For its information theory-based background, GTM
finds more separable map than PCA. The GTM problem is basically
Gaussian mixture model problem and a standard method to solve this
problem is Expectation-Maximization (EM) method.

We apply a novel optimization method, called Deterministic Annealing
(DA), to solve the local optimum problem which the original GTM can
suffer from.

Building
--------

Dependent Libraries are as follows

* MPI
* GNU Scientific Library (GSL)
* Parallel HDF5
* CMake


### Compile

1.  First make a directory for building and use cmake. We recommend to
use pre-defined script, config.sh, which will execute cmake with
proper parameters.

```
$ config.sh
USAGE : config.sh [options] <compiler = [gnu | intel | cray]>
```

2. Run make command. If you have any problem, you can see details as follows:

```
$ make VERBOSE=1
```

User Guide
----------

Once you finished building the program, you can run dagtm. The program takes the following options:

```
usage: dagtm [OPTIONS]
        -a number    : alpha (default: 0.99)
        -b name      : dataset name for beta (default: beta)
        -B name      : dataset name for label (default: lb)
        -c schedule  : cooling schedule [auto|exp|linear] (default : exp)
        -e number    : precision
        -f dataset   : dataset name for FI (default: FI)
        -h           : show help
        -i filename  : input HDF5 filename
        -I           : interpolation
        -j number    : maximum number of iterations
        -K number    : number of latent data K (eg, 100, 1k, 1M)
        -l           : log to file
        -L number    : latent dimension
        -m           : EM
        -M number    : number of model
        -n           : disable checkpointing
        -o filename  : output HDF5 filename
        -p           : log progress
        -P number    : checkpointing per num. loop
        -r filename  : restarting
        -s number    : seed
        -S number    : start temp
        -t dataset   : dataset name for T (default: T)
        -v number    : set verbose level
        -w prefix    : checkpointing file prefix
        -W dataset   : dataset name for W (default: W)
        -x dataset   : dataset name for X (default: X)
        -z PxQ       : define P-by-Q compute grid
```

For an example, if you want to process a data file (Oil.h5) in
8000(=20x20x20) latent point space by using 4 mpi processes, you can
type the followig command:

```
$ mpiexec -n 4 dagtm -i Oil.h5 -r -K 8000
```

In order to prepare input hdf5 file, you can use "text2h5" in `bin`
directory:

```
USAGE : text2h5 [OPTIONS] INFILE [IDFILE] [LBFILE]
OPTIONS:
  -d : dryrun
  -y : force y
```

Reference
---------

Please cite the following papers in any research that uses this
software:

```
@article{choi2010generative,
  title={Generative topographic mapping by deterministic annealing},
  author={Choi, Jong Youl and Qiu, Judy and Pierce, Marlon and Fox, Geoffrey},
  journal={Procedia Computer Science},
  volume={1},
  number={1},
  pages={47--56},
  year={2010},
  publisher={Elsevier}
}
@article{choi2011browsing,
  title={Browsing large-scale cheminformatics data with dimension reduction},
  author={Choi, Jong Youl and Bae, Seung-Hee and Qiu, Judy and Chen, Bin and Wild, David},
  journal={Concurrency and Computation: Practice and Experience},
  volume={23},
  number={17},
  pages={2315--2325},
  year={2011},
  publisher={Wiley Online Library}
}
```
Contact
-------

If you have any question regarding DA-GTM , please contact us one of
the following members:

* Jong Youl Choi (jychoi@indiana.edu, choij@ornl.gov)
* Judy Qiu (Professor at Indiana University)
* Geoffrey Fox (Professor at Indiana University)
