# torchscript

To run the Python programs, it should suffice to install the Python distribution of PyTorch from [pytorch.org](https://pytorch.org/).  Choose, for example, the stable build, your operating system, Conda or Pip, Python, and CPU.  It appears that setting the `LD_LIBRARY_PATH` as described in the next paragraph will interfere with Python operations, so avoid it or turn it off for the Python programs.

To run the Java or Scala programs, you need to have the libraries installed and accessible.  At [pytorch.org](https://pytorch.org/) choose, for example, the stable build, your operating system, LibTorch, C++/Java, and CPU.  Unzip the file on your hard drive and for Linux (and probably Mac), make sure the `LD_LIBRARY_PATH` environment variable has a value of the `lib` directory of the unzipped package, such as `/home/you/libtorch/lib`.  For Windows, make sure the `PATH` has an entry for that directory.
