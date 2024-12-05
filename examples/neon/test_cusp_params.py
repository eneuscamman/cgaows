import unittest
import numpy as np
import sys 
sys.path.append('../../src/')
import os
import pickle

import cusped_orbitals

class TestCusps(unittest.TestCase):

    ### minimum require input to define a system: neon atom ###
    options = {
    
        # nuclear positions IN BOHR!	
        "nuclei": """
         0.        0.          0.
        """,
    
        # nuclear charges
        "Z": np.array([
          [10.0,],
          ]),
    
        # basis for HF calculation
        # Our codebase only provides defaults for STO-3G, 6-31G, and 6-31G*
        # If you want to use a different basis, you will need to provide the gaussian exponents and coefficients. See basis_set_info.py for examples.
        "basis_type" : '6-31G*',
    
        # bool - if True, mocoeff defaults to pyscf HF calc 
        'HF_mocoeff' : False,
        
        # optional: molecular orbital coefficient matrix 
        "mocoeff": np.array([[ 9.952616267747e-01, -2.458189739119e-01, -5.590772595883e-04,  9.566506446026e-04, -9.211420754009e-04], 
                            [ 2.062742801093e-02,  5.455327197203e-01,  3.392906917793e-06, -5.866425229514e-06,  5.654113055432e-06],
                            [-3.234794432949e-03,  5.485106988603e-01,  1.671792338227e-05, -2.866686211369e-05,  2.760822382264e-05],
                            [ 7.190125867832e-04, -3.529235125998e-05,  5.600050881422e-01, -4.025255988413e-01,  2.835467343178e-02],
                            [-3.663626908777e-04,  1.802817647368e-05,  4.031785407746e-01,  5.561381630646e-01, -6.778152767020e-02],
                            [ 5.688965555387e-04, -2.795844101328e-05,  1.668174434462e-02,  7.155499512315e-02,  6.863223294134e-01],
                            [ 4.784606566072e-04, -2.348498742228e-05,  3.726505030665e-01, -2.678571500181e-01,  1.886837020301e-02],
                            [-2.437928582279e-04,  1.199669284181e-05,  2.682916445343e-01,  3.700772914904e-01, -4.510462658243e-02],
                            [ 3.785672525184e-04, -1.860470079498e-05,  1.110072132143e-02,  4.761564759710e-02,  4.567072098758e-01],
                            [ 3.232974861195e-19,  1.215827856550e-18,  3.153468200347e-17, -5.221523391040e-17, -5.790047264697e-18],
                            [-4.535628979862e-21,  4.596587833369e-19,  1.129560602093e-17, -1.205591451477e-17, -5.481765154472e-18],
                            [ 7.309212685378e-20,  2.800123802435e-17,  1.172862479265e-16, -2.464115000477e-17, -8.645314385405e-17],
                            [-4.541287844607e-20, -8.666870381585e-19,  2.328982942311e-18,  6.833045272545e-19,  3.974375836466e-18],
                            [-1.753530606695e-19,  8.541412926286e-18,  3.124387725049e-17,  2.422817253607e-17,  6.273831760844e-18],]),
    
    }

    # create and save all cusp params into text files in /examples/neon/test_data
    cusped_orbitals.get_cusped_orb_info(options,'test_data/',save_pkl=False)

    # TODO creat uniti test comparing each created *.txt file with the reference files (/examples/neon/*.txt to make sure they are all equiv
    def test_no(self):
      print("----> test no.txt")
      ref_no = np.loadtxt('no.txt')
      no = np.loadtxt('test_data/no.txt')
      self.assertEqual(ref_no, no, "numbor of orbitals correct")


if __name__ == '__main__':
    unittest.main()