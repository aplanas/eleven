#! /bin/sh

# make sure these are installed:
if false
then
    sudo zypper in python-devel blas-devel lapack-devel gcc-fortran
fi

# virtualenv env
# . ./env/bin/activate

pip install -U numpy scipy
pip install -U matplotlib

# pip install -U scikit-learn
pip install -U git+https://github.com/scikit-learn/scikit-learn.git#egg=scikit-learn

pip install -U ipython[all]

pip install -U git+https://github.com/gitpython-developers/gitdb.git#egg=gitdb

# https://pypi.python.org/packages/source/G/GitPython/GitPython-0.3.2.RC1.tar.gz
pip install -U git+https://github.com/gitpython-developers/GitPython.git#egg=GitPython

# Apply https://github.com/johnsca/GitPython/commit/db82455bd91ce00c22f6ee2b0dc622f117f07137
