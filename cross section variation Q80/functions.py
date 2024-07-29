import numpy as np
import torch as tn
from scipy.integrate import quad
from pdfpy import pdf
from constants import *
from math import pi

def f_s(x, tau_x, flavor, Q2):
    return 1/x *pdf.xfxQ2(flavor, x, Q2) * 1/tau_x *pdf.xfxQ2(-flavor, tau_x, Q2) + \
           1/tau_x *pdf.xfxQ2(flavor, tau_x, Q2) *1/x * pdf.xfxQ2(-flavor, x, Q2)

def num_derivative(func, x, h=1e-15, *args):
    return (func(x + h, *args) - func(x , *args)) / (h)

def f_prime_s(x, tau_x, flavor, Q2):
    f_f_tau_x_prime = num_derivative(lambda t: 1/t *pdf.xfxQ2(flavor, t, Q2), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t:1/t *pdf.xfxQ2(-flavor, t, Q2), tau_x)
    
    return (1/x * pdf.xfxQ2(flavor, x, Q2) *f_fbar_tau_x_prime + \
           1/x *f_f_tau_x_prime *pdf.xfxQ2(-flavor, x, Q2))
           

def sigma_hat_prime(x, tau_x, C, p1, p2, flavor, Q2, n):
    
    # Perform the contraction with c

    term1 = n* f_s(x, tau_x, flavor, Q2)
    
    term2 = ( (2 / s) * (1 + x / tau_x) *
             (tn.dot(p1, tn.mv(C, p2)) + tn.dot(p2, tn.mv(C, p1)) +
              tn.dot(p1, tn.mv(C, p1)) + tn.dot(p2, tn.mv(C, p2)))) * f_s(x, tau_x, flavor, Q2)

    
    term3 = (2 / s) *(x * tn.dot(p1, tn.mv(C, p1)) + tau_x * tn.dot(p1, tn.mv(C, p2)) +
             tau_x * tn.dot(p2, tn.mv(C, p1)) + x * tn.dot(p2, tn.mv(C, p2)))* f_prime_s(x, tau_x, flavor, Q2)
    
    return term1, term2+term3
    

def integrate_sigma_hat_prime(tau, C, p1, p2, flavor, Q2, n):
    def integrand1(x):
        tau_x = tau / x
        term1, _ = sigma_hat_prime(x, tau_x, C, p1, p2, flavor, Q2, n)
        return  term1 * tau_x

    def integrand2(x):
        tau_x = tau / x
        _, term2_plus_term3 = sigma_hat_prime(x, tau_x, C, p1, p2, flavor, Q2, n)
        return term2_plus_term3 * tau_x

    result1, _ = quad(integrand1, tau, 1)  
    result2, _ = quad(integrand2, tau, 1)  

    return result1+s*result2

    
def term_1(Q2, e_f):
    return e_f**2 / (2*Q2**2)
    
def term_2(Q2, e_f, g, p1, p2):
    return (((1 - (m_Z**2 / Q2)) / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2)) *
            (1 - 4 * sin2th_w) / (4 * sin2th_w * (1- sin2th_w ))* e_f * g)
            
def term_3(Q2, e_f, g, p1, p2):
    return (1 / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2) * 
            (1 + (1 - 4 * sin2th_w)**2) / (32 * sin2th_w**2 * (1-sin2th_w)**2)) * g**2
            
def d_sigma(Q2, CL, CR, p1, p2, n, quark_couplings):
    tau = Q2 / s
    d_sigmaL = 0
    d_sigmaR = 0

    for flavor, e_f, g_fR, g_fL in quark_couplings:
            integral1 = integrate_sigma_hat_prime(tau, CL, p1, p2, flavor, Q2, n)
            integral2 = integrate_sigma_hat_prime(tau, CR, p1, p2, flavor, Q2, n)
    
            d_sigmaL += 4 * np.pi * alpha**2 / (3 * Nc) * (
                        term_1(Q2, e_f) + term_2(Q2, e_f, g_fL, p1, p2) + 
                        term_3(Q2, e_f, g_fL, p1, p2)
                        ) * integral1
    
            d_sigmaR += 4 * np.pi * alpha**2 / (3 * Nc) * (
                        term_1(Q2, e_f) + term_2(Q2, e_f, g_fR, p1, p2) + 
                        term_3(Q2, e_f, g_fR, p1, p2)
                        ) * integral2
            
    
    
    return 0.38*1e9*(d_sigmaL+d_sigmaR) # conversation from GeV-2 to Pb
