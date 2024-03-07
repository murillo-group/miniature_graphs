## Modified Lee-More Model (MLM) for Electronic Transport

MLM is a Python library that evaluates the Lee-More (LM) electrical conductivity model. Currently (as of April 2022) the MLM only calculates the electrical conductivity for an unmagnetized plasma. 

The original LM paper is [here](https://aip.scitation.org/doi/10.1063/1.864744).

### What is it? 

The MLM is a variant of the original LM model that makes slightly different choices for parameters.  Briefly, choices for undefined parameters (e.g., mean distance between particles) and inconsistent parameters (e.g., using an effective temperature) were made. 

Although the MLM is a standalone library, it was originally developed in the context of discrepancy learning in which the MLM is the base model and conductivity data (originally from Katsouros and DeSilva; [here](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.57.5945)) was used to learn the discrepancy using radial basis function neural networks. That original work is cited below, and many more details can be found there.

Described here is just the base MLM model as a stand-alone library. 

### How do I use it? 

The first step is to install the libray using
```
pip install MLM_Transport
```

Next, import the library into your Python code with
```
import MLM
```
and ensuring that it is in your path.

To compute an electrical conductivity with a known element with nuclear charge _Z_, atomic number _A_, mass density ρ in g/cc and temperature _T_ in eV, use
```
elect_cond = MLM.sigma(Z, A, rho, T)
elect_cond_2T = MLM.sigma(Z, A, rho, T_e, T_i)
```
It's that easy! Output units are _S/m_. 


### Are there other functions?

Yes. To compute transport properties, we need several intermediate quantities that are also of use in other contexts. These are:
* `zbar`: Thomas-Fermi mean ionization state <Z>,
* `effective_temperature`: compute the effective electron temperature using Fermi integrals,
* `Ichimaru_chem_pot`: fit to the ideal chemical potential as a function of degeneracy, βμ(θ), as given by Ichimaru,
* `FD_int`: Fermi integral of any order computed by series summation,
* `melt_temperature`: estimate of the melting temperature of any element at any density.

The Thomas-Fermi ionization model can be called with:
```
Z = 13
A = 26
rho = 2.7
T = 10
MLM.zbar(Z, A, rho, T)
3.018990389147884
```
  
Effective temperatures require the electron temperature and η=βμ:
```
MLM.effective_temperature(10.0, 0.5)
13.832742338864064 
```
  
  
For the chemical potential:
```
print(MLM.Ichimaru_chem_pot(2.7183))
-1.7226056272828079
```
  
There are three Fermi integral routines:
1. order 0, (exact),
2. order -1 (exact), 
3. general purpose (any order, any η).

Calling the integrals requires η=βμ and the order; you can optionally control the accuracy for the general purpose case:
```
FD_val_0p5 = MLM.FD_int(-3.14, 1/2)
FD_val_0 = MLM.FD_int_0(2)    # order 0
FD_val_m1 = MLM.FD_int_m1(2)  # order -1
FD_val_0p5_acc = MLM.FD_int(-3.14, 1/2, EPS=1e-7)
```
The convention used has no Γ function prefactor. See the source code for other options.

Estimating the melting temperature can be done by
```
Z = 13
ion_density = 2.375/(1.6737e-24*27)
print(MLM.melt_temperature(Z, ion_density))
print(MLM.melt_temperature(Z, ion_density, original=True))
0.08442852430440988
0.08338125825141993
```
Here, `original` uses the fit parameters from LM. 
  
  
### What's next?

The next version of MLM will include thermal conductivity. Magnetized plasmas for both transport coefficients are next, followed by related transport coefficients (e.g., Hall, Nernst, thermoelectric power (Seebeck), Leduc-Righi, Ettinghausen).


### Where I can find the data? 

You might also be interested in the experimental data that motivated this project. I have compiled the Katsouros-DeSilva data [here](https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database/tree/master/database/DeSilvaKatsouros) and also provide a Jupyter notebook that allows for quick start with the data. 


### How do I cite this? 

The original version of this work, including more details, appeared in:
* Data-driven Electrical Conductivities of Dense Plasmas, Michael S. Murillo, Frontiers in Physics, 2022