#! /bin/sh

# virtualenv env
# . ./env/bin/activate

pip install -U numpy scipy
pip install -U matplotlib

# pip install -U scikit-learn
pip install -U git+https://github.com/scikit-learn/scikit-learn.git#egg=scikit-learn

pip install -U ipython[all]

pip install -U https://pypi.python.org/packages/source/G/GitPython/GitPython-0.3.2.RC1.tar.gz
