# torchscript

To run the Python programs, it should suffice to install the Python distribution of PyTorch from [pytorch.org](https://pytorch.org/).  Choose, for example, the stable build, your operating system, Conda or Pip, Python, and CPU.  It appears that setting the `LD_LIBRARY_PATH` as described in the next paragraph will interfere with Python operations, so avoid it or turn it off for the Python programs.

To run the Java or Scala programs, you need to have the libraries installed and accessible.  At [pytorch.org](https://pytorch.org/) choose, for example, the stable build, your operating system, LibTorch, C++/Java, and CPU.  Download the packange and unzip the file on your hard drive.

* Linux

  Make sure the `LD_LIBRARY_PATH` environment variable includes the the `lib` directory of the unzipped package, such as `/home/you/libtorch/lib`.  The operating system will use this to load `*.so` files and also to set the value of `java.library.path`.

  ```bash
  $ export LD_LIBRARY_PATH=/home/you/libtorch/lib`
   ```
  
* Windows

  Make sure the `PATH` environment variable includes the `lib` directory of the unzipped package, such as `D:\Users\you\libtorch\lib`.  The operating system will use this to load `*.dll` files and also to set the value of `java.library.path`.

  For the regular command prompt:
  ```bat
  > set PATH=D:\Users\you\libtorch\lib;%PATH%`
  ```
  
  For PowerShell:
  ```powershell
  PS> $env:PATH="D:\Users\you\libtorch\lib;" + $env:PATH`
  ```

* Mac

  This is problematic.  Java [apparently](https://help.mulesoft.com/s/article/Variables-LD-LIBRARY-PATH-DYLD-LIBRARY-PATH-are-ignored-on-MAC-OS-if-System-Integrity-Protect-SIP-is-enable) does not have access to the `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` variables and cannot use them to build `java.library.path` or use them in `sbt` so that a torchscript program will run from there.  So far the only way the example program has worked is by being called directly from Java after `sbt dist` with all the aforementioned variables having been set manually.

  ```bash
  $ export LD_LIBRARY_PATH=/home/you/libtorch/lib
  $ export DYLD_LIBRARY_PATH=/home/you/libtorch/lib
  $ java -Djava.library.path=/home/you/libtorch/lib ...
  ```