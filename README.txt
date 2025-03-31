        
----------------------------------------------------------------------
        Cusping Gaussian Atomic Orbitals with Slaters (CGAOWS)
----------------------------------------------------------------------

###############################################
#### Calculating cusp parameters in PYTHON ####
###############################################

1.) To test the calculation of the cusp parameters via 
    python in the base cgaows/ directory run:

    >> python -m examples.neon.test_cusp_params

    this generates, and saves to files in the output path 
    defined in examples/neon/test_cusp_params.py, all the 
    cusping parameters that would by provided to a VMC
    sampling framework

    The unittest compares the newly calculated set of cusp 
    parameters (cusp_coeff_mat.txt) to the reference result 
    in example/neon/cusp_coeff_mat.txt. All txt files could 
    be compare, however that is not currently implemented

    Note: this step uses multiprocessing to calculate
          the cusp parameters. With multiprocessing this 
          step takes ~10s on a desktop

2.) An example input for neon is provided in 
    examples/neon/input.py, containing the minimum 
    required information needed
    (ie nuclear positions, nuclear charges, basis set)

    To reproduce the reference data used to compare the 
    result of the unittest in 1.), in examples/neon/ run:

    >> python input.py

    to generate the cusp information as txt 
    files which will then be read into  C++ code. 

    # Not sure if this is True?? - TKQ

###############################################
############## Running C++ Code ###############
###############################################

3.) We have provided an example cpp file in 
    examples/neon/example_Ne.cpp that demonstrates how 
    the cusp functionality may be integrated into 
    a VMC code. 

    "example_Ne.cpp" creates a CuspedGaussians object and
    calls the relevant functions "evaluate_orbs" and 
    "evaluate_derivs." These functions take the addresses 
    of the electron and nuclear positions in addition to 
    the matrices/pointers to matrices that are populated 
    with the cusped atomic orbital evaluations and 
    derivative evaluations for a given electron 
    configuration.


4.) To build build CGAOWS and make executable (in cgaows-v1.0):

    >> mkdir build && cd build
    >> cmake -DCMAKE_INSTALL_PREFIX=/path/to/desired/install/location ..
    >> make
    >> make test
    >> make install
    >> ./run_example   # note this example is in /examples/neon, 
                       # example_neon.cpp and *.txt parameter 
                       # files must be in the same dir

The python part of CGAOWS will be installed under the directory
python_module within your install location.  Don't forget to put
that directory in your PYTHONPATH.

5.) The output of ./run_example can then be used for the Markov chain 
    propagation and local energy evaluation in a VMC calculation. 

Note: MAKE SURE to use the transformed molecular orbital coefficient
matrix that is printed out from the cusped_orbital.py for all VMC 
calculationss. This takes the orthogonalization of the higher-order 
s-type AOs into account.
