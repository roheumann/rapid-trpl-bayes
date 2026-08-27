##Constants
import numpy as np
Q = 1.602176462e-19 ##                     [As], elementary charge
K = 1.38064852e-23  #;                     % [(m^2)kg(s^-2)(K^-1)], Boltzmann constant
T = 300     #;                             % [K], temperature
VT = K*T/Q  #;                             % [V], ~25.8mV thermal voltage at 300K
H_JS = 6.62606876e-34                         #% Planck's constant(Js)
H_EVs = 4.135667662e-15                       #% Planck's constant(eVs)
C = 299792458                                 #% Lichtgeschwindigkeit c (m/s)

#### Contants to give
NC = 2.21359e18                                 #; % effective DOS in CB
NV = 2.21359e18                                  #; % effective DOS in VB

QFLS_STEPS = 256 #number of time steps in the tau vs t curve 
DECAY_MAGNITUDE = 12    #Look at maximum 12 orders of magnitude difference in decay

T_ARR = [1e-11, 1e5]  #in seconds