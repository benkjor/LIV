import numpy as np
import torch as tn
from scipy.integrate import quad
from pdfpy import pdf
from constants import *
from math import pi

factor = 4 * np.pi * alpha**2 / (3 * Nc)

def f_s(x, tau, flavor, Q2):
    tau_x = tau / x
    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_flavor_tau_x = pdf.xfxQ2(flavor, tau_x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    pdf_anti_flavor_tau_x = pdf.xfxQ2(-flavor, tau_x, Q2)

    term1 = (1 / x) * pdf_flavor_x * (1/tau_x) * pdf_anti_flavor_tau_x
    term2 = (1/tau_x) * pdf_flavor_tau_x * (1 / x) * pdf_anti_flavor_x
    
    return term1 + term2

def num_derivative(func, x, h=1e-7 ,*args):
    return (func(x + h, *args) - func(x , *args)) / (h)

def f_prime_s(x, tau, flavor, Q2):
    tau_x = tau/x
    f_f_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(flavor, t, Q2), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(-flavor, t, Q2), tau_x)

    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    
    return 1/x * pdf_flavor_x * f_fbar_tau_x_prime + \
           1/x * f_f_tau_x_prime * pdf_anti_flavor_x

def sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2):
    tau_x = tau/x

    f_s_val = f_s(x, tau, flavor, Q2)
    f_prime_s_val = f_prime_s(x, tau, flavor, Q2)
    
    # Efficiently handle the contraction with non-zero elements of C
    contraction_p1p1 = tn.einsum('mn,m,n->', C, p1, p1)
    contraction_p1p2 = tn.einsum('mn,m,n->', C, p1, p2)
    contraction_p2p2 = tn.einsum('mn,m,n->', C, p2, p2)
    
    term1 = f_s_val
    
    term2 = (2 / s) * (1 + x / tau_x) * (contraction_p1p1 + 2 * contraction_p1p2 + contraction_p2p2) * f_s_val
    
    term3 = (2 / s) * (x * contraction_p1p1 + 2 * tau_x * contraction_p1p2 + x * contraction_p2p2) * f_prime_s_val
    
    return term1, term2 + term3

def integrate_sigma_hat_prime(tau, C, p1, p2, flavor, Q2):
    def integrand1(x):
        tau_x = tau / x
        term1, _ = sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2)
        return term1 * tau_x

    def integrand2(x):
        tau_x = tau / x
        _, term2_plus_term3 = sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2)
        return term2_plus_term3 * tau_x

    result1, _ = quad(integrand1, tau, 1)
    result2, _ = quad(integrand2, tau, 1)

    return result1 + s * result2

def term_1(Q2, e_f):
    return e_f**2 / (2*Q2**2)
    
def term_2(Q2, e_f, g):
    return (((1 - (m_Z**2 / Q2)) / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2)) *
            (1 - 4 * sin2th_w) / (4 * sin2th_w * (1- sin2th_w ))* e_f * g)
            
def term_3(Q2, e_f, g):
    return (1 / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2) * 
            (1 + (1 - 4 * sin2th_w)**2) / (32 * sin2th_w**2 * (1-sin2th_w)**2)) * g**2

def summation_terms(Q2, e_f, g):
    return term_1(Q2, e_f) + term_2(Q2, e_f, g) + term_3(Q2, e_f, g)

def d_sigma(Q2, CL, CR, p1, p2, quark_couplings):
    tau = Q2 / s
    d_sigmaL = 0
    d_sigmaR = 0

    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral1 = integrate_sigma_hat_prime(tau, CL, p1, p2, flavor, Q2)
        integral2 = integrate_sigma_hat_prime(tau, CR, p1, p2, flavor, Q2)
        sum_terms_L = summation_terms(Q2, e_f, g_fL)
        sum_terms_R = summation_terms(Q2, e_f, g_fR)

        d_sigmaL += factor * sum_terms_L * integral1
        d_sigmaR += factor * sum_terms_R * integral2

    return 0.38 * 1e9 * (d_sigmaL + d_sigmaR)  # Conversion from GeV-2 to Pb

def integrate_sigma_hat_prime_sm(tau, flavor, Q2):
    def integrand1(x):
        return f_s(x, tau, flavor, Q2) * tau / x
    
    result, _ = quad(integrand1, tau, 1)
    return result

def d_sigma_sm(Q2, quark_couplings):
    tau = Q2 / s
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(tau, flavor, Q2)
        
        sum_terms_L = summation_terms(Q2, e_f, g_fL)
        sum_terms_R = summation_terms(Q2, e_f, g_fR)
        
        d_sigma += factor * (sum_terms_L + sum_terms_R) * integral

    return 0.38 * 1e9 * d_sigma  # Conversion from GeV^-2 to Pb
