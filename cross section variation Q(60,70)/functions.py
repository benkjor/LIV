import numpy as np
import torch as tn
from scipy.integrate import quad
from numdifftools import Derivative as num_derivative
from pdfpy import pdf
from constants import *
from math import pi

def f_s(x, tau_x, flavor, Q):
    return 1/x *pdf.xfxQ(flavor, x, Q) * 1/tau_x *pdf.xfxQ(-flavor, tau_x, Q) + \
           1/tau_x *pdf.xfxQ(flavor, tau_x, Q) *1/x * pdf.xfxQ(-flavor, x, Q)

def num_derivative(func, x, h=1e-15, *args):
    return (func(x + h, *args) - func(x , *args)) / (h)

def f_prime_s(x, tau_x, flavor, Q):
    f_f_tau_x_prime = num_derivative(lambda t: 1/t *pdf.xfxQ(flavor, t, Q), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t:1/t *pdf.xfxQ(-flavor, t, Q), tau_x)
    
    return (1/x * pdf.xfxQ(flavor, x, Q) *f_fbar_tau_x_prime + \
           1/x *f_f_tau_x_prime *pdf.xfxQ(-flavor, x, Q))
           

def sigma_hat_prime(x, tau_x, C, p1, p2, flavor, Q, n):
    
    # Perform the contraction with c
    
    term1 = (n + (2 / s) * (1 + x / tau_x) *
             (tn.dot(p1, tn.mv(C, p2)) + tn.dot(p2, tn.mv(C, p1)) +
              tn.dot(p1, tn.mv(C, p1)) + tn.dot(p2, tn.mv(C, p2)))) * f_s(x, tau_x, flavor, Q)

    
    term2 = (2 / s) *(x * tn.dot(p1, tn.mv(C, p1)) + tau_x * tn.dot(p1, tn.mv(C, p2)) +
             tau_x * tn.dot(p1, tn.mv(C, p2)) + x * tn.dot(p2, tn.mv(C, p2)))* f_prime_s(x, tau_x, flavor, Q)
    
    return term1 + term2
    
def integrate_sigma_hat_prime(tau,C, p1, p2,flavor, Q, n):
    def integrand(x):
        tau_x = tau/x
        return sigma_hat_prime(x, tau, C, p1, p2, flavor, Q, n) * tau_x
    
    result, error = quad(integrand, tau, 1)
    return result
    
def term_1(Q2, e_f):
    return 0.389* 1e9* e_f**2 / (2*Q2**2)
    
def term_2(Q2, e_f, g, p1, p2):
    return 0.389* 1e9*(((1 - (m_Z**2 / Q2)) / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2)) *
            (1 - 4 * sin2th_w) / (4 * sin2th_w * (1- sin2th_w ))* e_f * g)
            
def term_3(Q2, e_f, g, p1, p2):
    return 0.389* 1e9*(1 / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2) * 
            (1 + (1 - 4 * sin2th_w)**2) / (32 * sin2th_w**2 * (1-sin2th_w)**2)) * g**2
            
def d_sigma(Q2, CL, CR, p1, p2, n, quark_couplings):
    tau = Q2 / s
    d_sigma = 0
    d_sigmaL = 0
    d_sigmaR = 0

    for flavor, e_f, g_fR, g_fL in quark_couplings:
            integral1 = (13e3)**2 *integrate_sigma_hat_prime(tau, CL, p1, p2, flavor, np.sqrt(Q2), n)
            integral2 = (13e3)**2 *integrate_sigma_hat_prime(tau, CR, p1, p2, flavor, np.sqrt(Q2), n)
    
            d_sigmaL += 4 * np.pi * alpha**2 / (3 * Nc) * (
                        term_1(Q2, e_f) + term_2(Q2, e_f, g_fL, p1, p2) + 
                        term_3(Q2, e_f, g_fL, p1, p2)
                        ) * integral1
    
            d_sigmaR += 4 * np.pi * alpha**2 / (3 * Nc) * (
                        term_1(Q2, e_f) + term_2(Q2, e_f, g_fR, p1, p2) + 
                        term_3(Q2, e_f, g_fR, p1, p2)
                        ) * integral2
            
    
    
    return d_sigmaL+d_sigmaR, d_sigmaL, d_sigmaR
