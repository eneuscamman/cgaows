#ifndef CUSPED_GAUSSIANS_H
#define CUSPED_GAUSSIANS_H

#include <iomanip>
#include <typeinfo>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

class CuspedGaussians {

  protected:
    // path to the cusping parameters to be read in
    std::string m_paramPath;

    // number of orbitals
    int m_no;

    // number of nuclei
    int m_nn;

    // charges of nuclei
    std::vector<double> m_Z;
 
    // number of electrons
    int m_ne;
  
    // maximum number of gaussian primitives per basis function (3 for STO-3G and 6 for 6-31G or 6-31G*)
    int m_ng; 

    // array of basis set centers (indices of nuclei)
    std::vector<int> m_bf_cen;

    // array that holds each basis function type (using 5 d-orbitals as in Pyscf) 
    // (1s = 0, 2s,3s,..,ns = 1, 2px = 2, 2py = 3, 2pz = 4, 3dxy = 5, 3dyz = 6, 3dz^2 = 7, 3dxz = 8, 3dx^2-y^2 = 9) 
    std::vector<int> m_bf_type;

    // array of basis set exponents
    std::vector<double> m_bf_exp;

    // array of basis set coefficients
    std::vector<double> m_bf_coeff;

    // matrix of gaussian cusp cutoff radii
    std::vector<double> m_cusp_radii_mat;

    // matrix of gaussian cusp parameters a_0
    std::vector<double> m_cusp_a0_mat;

    // vector of assigning which orbs get orthogonalized (val based on orb_ind+1 they get orth against)
    std::vector<int> m_orth_orb;

    // matrix of projection values for orthogonalized orbs
    std::vector<double> m_proj_mat;

    // tensor of P basis function coefficients (# nuc x # AO x # nbf)
    std::vector<double> m_cusp_coeff_mat;

    // array of orders of n for Pn basis functions
    std::vector<double> m_n_vec;

    // template for reading in txt files into members
    template <typename T> 
    void read_in_file(const std::string & filename, T & data) {
      // open file
      std::ifstream file(filename);
      // throw error if unable to open
      if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << " for reading.\n";
      }
      // read and store to vector
      T value;
      file >> value; 
      data = value;
      // close file
      file.close();
    }
 
    // template for reading in txt files into members
    template <typename T> 
    void read_in_file(const std::string & filename, std::vector<T> & data) {
      // open file
      std::ifstream file(filename);
      // throw error if unable to open
      if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << " for reading.\n";
      }
      // read and store to vector
      T value;
      while (file >> value) {
        data.push_back(value);
      }
      // close file
      file.close();
    } 
  
  public:

    // Constructor
    CuspedGaussians(const std::string& paramPath = ".") : m_paramPath(paramPath)
    {

      // read in txt files here
      
      // number of orbs, m_no 
      this->read_in_file(m_paramPath+"/no.txt", m_no);    
      /*
      std::cout << "m_no from reading txt: " << m_no << std::endl;
      */

      // number of nuclei, m_nn 
      this->read_in_file(m_paramPath+"/nn.txt", m_nn);
      /*
      std::cout << "m_nn from reading txt: " << m_nn << std::endl;
      */
      
      // charges Z of nuclei, m_Z 
      this->read_in_file(m_paramPath+"/Z.txt", m_Z);
      /*
      std::cout << "Z" << std::endl;
      for (int i = 0; i < m_Z.size(); i++) {
        std::cout << m_Z[i] << std::endl;
      }
      */

      // number of electrons, m_ne 
      this->read_in_file(m_paramPath+"/ne.txt", m_ne);
      /*
      std::cout << "m_ne from reading txt: " << m_ne << std::endl;
      */
      
      // max number of gaussians per bf, m_ng 
      this->read_in_file(m_paramPath+"/ng.txt", m_ng);    
      /*
      std::cout << "m_ng from reading txt: " << m_ng << std::endl;
      */

      // bf centers, m_bf_cen 
      this->read_in_file(m_paramPath+"/bf_cen.txt", m_bf_cen);
      /*
      std::cout << "bf cen vec" << std::endl;
      for (int i = 0; i < m_bf_cen.size(); i++) {
        std::cout << m_bf_cen[i] << std::endl;
      }
      */

      // bf types, m_bf_type 
      this->read_in_file(m_paramPath+"/bf_type.txt", m_bf_type);
      /*
      std::cout << "bf type vec" << std::endl;
      for (int i = 0; i < m_bf_type.size(); i++) {
        std::cout << m_bf_type[i] << std::endl;
      }
      */

      // bf exponents, m_bf_exp 
      this->read_in_file(m_paramPath+"/bf_exp.txt", m_bf_exp);
      /*
      std::cout << "bf exp vec" << std::endl;
      for (int i = 0; i < m_bf_exp.size(); i++) {
        std::cout << m_bf_exp[i] << std::endl;
      }
      */
    
      // bf coeffs, m_bf_coeff 
      this->read_in_file(m_paramPath+"/bf_coeff.txt", m_bf_coeff);
      /*
      std::cout << "bf coeff vec" << std::endl;
      for (int i = 0; i < m_bf_coeff.size(); i++) {
        std::cout << m_bf_coeff[i] << std::endl;
      }
      */
      
      // cusp radii mat, m_cusp_radii_mat 
      this->read_in_file(m_paramPath+"/cusp_radii_mat.txt", m_cusp_radii_mat);
      /*
      std::cout << "cusp_radii_mat" << std::endl;
      for (int i = 0; i < m_cusp_radii_mat.size(); i++) {
        std::cout << m_cusp_radii_mat[i] << std::endl;
      }
      */

      // cusp a0 mat, m_cusp_a0_mat 
      this->read_in_file(m_paramPath+"/cusp_a0_mat.txt", m_cusp_a0_mat);
      /*
      std::cout << "cusp_a0_mat" << std::endl;
      for (int i = 0; i < m_cusp_a0_mat.size(); i++) {
        std::cout << m_cusp_a0_mat[i] << std::endl;
      }
      */

      // orth orb vector, m_orth_orb 
      this->read_in_file(m_paramPath+"/orth_orb.txt", m_orth_orb);
      /*
      std::cout << "orth_orb" << std::endl;
      for (int i = 0; i < m_orth_orb.size(); i++) {
        std::cout << m_orth_orb[i] << std::endl;
      }
      */

      // projection matrix for the orthogonalized orbs, m_proj_mat 
      this->read_in_file(m_paramPath+"/proj_mat.txt", m_proj_mat);
      /*
      std::cout << "proj_mat" << std::endl;
      for (int i = 0; i < m_proj_mat.size(); i++) {
        std::cout << m_proj_mat[i] << std::endl;
      }
      */
      
      // cusp coeff mat, m_cusp_coeff_mat 
      this->read_in_file(m_paramPath+"/cusp_coeff_mat.txt", m_cusp_coeff_mat);
      /*
      std::cout << "cusp_coeff_mat" << std::endl;
      for (int i = 0; i < m_cusp_coeff_mat.size(); i++) {
        std::cout << m_cusp_coeff_mat[i] << std::endl;
      }
      */

      // vector of Pn basis function polynomial orders, m_n_vec 
      this->read_in_file(m_paramPath+"/n_vec.txt", m_n_vec);
      /*
      std::cout << "n_vec" << std::endl;
      for (int i = 0; i < m_n_vec.size(); i++) {
        std::cout << m_n_vec[i] << std::endl;
      }
      */

    } // end constructor

    // function to evaluate s-type Gaussian orbital
    double GTO_s(double r, int ao_ind);
    // for the orthogonalized s orbital
    double GTO_s(double r, int ao_ind, int core_ao_ind, double proj);
    
    // evaluate 1st derivatives of s-type GTO
    void d1_GTO_s(const double r, int ao_ind, const double (&delta)[3], double (&d1_vec)[3]);
    // for the orthogonalized s-type GTO
    void d1_GTO_s(const double r, int ao_ind, int core_ao_ind, double proj, const double (&delta)[3], double (&d1_vec)[3]);
    
    // evaluate 2nd derivatives of s-type GTO
    void d2_GTO_s(const double r, int ao_ind, const double (&delta)[3], double (&d2_vec)[3]);
    // for the orthogonalized s-type GTO
    void d2_GTO_s(const double r, int ao_ind, int core_ao_ind, double proj, const double (&delta)[3], double (&d2_vec)[3]);
    
    // function to evaluate p-type Gaussian orbital
    double GTO_p(double r, double (&dist_xi)[3], int ao_ind, int ao_type);

    // evaluate 1st derivatives of p-type orbital
    void d1_GTO_p(const double r, int ao_ind, const double (&delta)[3], int ao_type, double (&d1_vec)[3]);

    // evaluate 2nd derivatives of p-type orbital
    void d2_GTO_p(const double r, int ao_ind, const double (&delta)[3], int ao_type, double (&d2_vec)[3]);
    
    // function to evaluate d-type Gaussian orbital
    double GTO_d(double r, double (&dist_xi)[3], int ao_ind, int ao_type);
 
    // evaluate 1st derivatives of d-type GTO
    void d1_GTO_d(const double dist, double (&delta)[3], int ao_ind, int ao_type, double (&d1_vec)[3]);   
    
    // evaluate 2nd derivatives of d-type GTO
    void d2_GTO_d(const double dist, double (&delta)[3], int ao_ind, int ao_type, double (&d2_vec)[3]);   
 
    // function to evaluate switching function
    double b_func(double rc, double nuc_dist);

    // get 1st derivatives of switching function 
    void d1_b_func(double rc, double nuc_dist, int i, int n, double (&diff)[3], double (&d1_b)[3]);

    // get 2nd derivatives of switching function 
    void d2_b_func(double rc, double nuc_dist, int i, int n, double (&diff)[3], double (&d2_b)[3]);

    // function to evaluate slater function f = a0 * exp(-zeta * |r-R|)
    double slater_func(double a0, double zeta, double r);

    // get 1st derivatives of slater function
    void d1_slater_func(double a0, double zeta, double r, int i, int n, double (&diff)[3], double (&d1_Q_vec)[3]);

    // get 2nd derivatives of slater function
    void d2_slater_func(double a0, double zeta, double r, int i, int n, double (&diff)[3], double (&d2_Q_vec)[3]);

    // function to evaluate higher order Slater-type functions
    double Pn_func(double b, double Q, double orb_total, double nuc_dist, double n, double rc);

    // get 1st derivatives of higher order Slater-type functions
    void d1_Pn_func(double &orb_total, double &b_val, double &Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double &nuc_dist, double (&diff)[3], double &n, double &qn, double (&d1_P_vec)[3], double &rc);

    // get 2nd derivatives of higher order Slater-type functions
    void d2_Pn_func(double &orb_total, double &b_val, double &Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double (&der2_total)[3], double (&d2_b_func)[3], double (&d2_slater_s_cusp)[3], double &nuc_dist, double (&diff)[3], double &n, double &qn, double (&d2_P_vec)[3], double &rc);   

    // get 1st derivatives of (1-b)X term
    void d1_one_minus_b(double &orb_total, double &b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&d1_Pn_vec)[3]);

    // get 2nd derivatives of (1-b)X term
    void d2_one_minus_b(double &orb_total, double &b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&der2_total)[3], double (&d2_b_vec)[3], double (&d2_Pn_vec)[3]);
 
    // function to evaluate the cusped atomic orbital values
    // Populates (num elec) x (num orb) matrix with orbital evaluations
    void evaluate_orbs(std::vector<double> &e_pos, std::vector<double> &n_pos, std::vector<double> &AO_mat);

    // function to evaluate the orbitals and 1st and 2nd derivatives 
    // Populates (num elec) x (num orb) matrices with x, y, z 1st and 2nd derivatives
    void evaluate_derivs(std::vector<double> &e_pos, std::vector<double> &n_pos, double **der1, double **der2);
 
};


#endif
