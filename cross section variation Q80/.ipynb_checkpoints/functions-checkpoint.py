import numpy as np
import torch as tn
from scipy.integrate import quad
from pdfpy import pdf
from constants import *
from math import pi
import multiprocessing as mp

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

# Ensure tensors are on the GPU if available (to be done in the main process)
device = None

def init_device():
    global device
    device = 'cuda' if tn.cuda.is_available() else 'cpu'

factor = 4 * np.pi * alpha**2 / (3 * Nc)

def f_s(x, tau, flavor, Q2):
    tau_x = tau / x
    pdf_flavor_x = tn.tensor(pdf.xfxQ2(flavor, x, Q2), device=device)
    pdf_flavor_tau_x = tn.tensor(pdf.xfxQ2(flavor, tau_x, Q2), device=device)
    pdf_anti_flavor_x = tn.tensor(pdf.xfxQ2(-flavor, x, Q2), device=device)
    pdf_anti_flavor_tau_x = tn.tensor(pdf.xfxQ2(-flavor, tau_x, Q2), device=device)

    term1 = (1 / x) * pdf_flavor_x * (1/tau_x) * pdf_anti_flavor_tau_x
    term2 = (1/tau_x) * pdf_flavor_tau_x * (1 / x) * pdf_anti_flavor_x
    
    return term1 + term2

def num_derivative(func, x, h=1e-7, *args):
    x = tn.tensor(x, device=device)
    return (func(x + h, *args) - func(x, *args)) / h

def f_prime_s(x, tau, flavor, Q2):
    tau_x = tau/x
    f_f_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(flavor, t.item(), Q2), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(-flavor, t.item(), Q2), tau_x)

    pdf_flavor_x = tn.tensor(pdf.xfxQ2(flavor, x, Q2), device=device)
    pdf_anti_flavor_x = tn.tensor(pdf.xfxQ2(-flavor, x, Q2), device=device)
    
    return 1/x * pdf_flavor_x * f_fbar_tau_x_prime + \
           1/x * f_f_tau_x_prime * pdf_anti_flavor_x

def sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2):
    x = tn.tensor(x, device=device)
    tau_x = tau/x

    f_s_val = f_s(x, tau, flavor, Q2)
    f_prime_s_val = f_prime_s(x, tau, flavor, Q2)
    
    # Efficiently handle the contraction with non-zero elements of C
    C = tn.tensor(C, device=device)
    p1 = tn.tensor(p1, device=device)
    p2 = tn.tensor(p2, device=device)
    
    contraction_p1p1 = tn.einsum('mn,m,n->', C, p1, p1)
    contraction_p1p2 = tn.einsum('mn,m,n->', C, p1, p2)
    contraction_p2p2 = tn.einsum('mn,m,n->', C, p2, p2)
    
    term1 = f_s_val
    
    term2 = (2 / 1) * (1 + x / tau_x) * (contraction_p1p1 + 2 * contraction_p1p2 + contraction_p2p2) * f_s_val
    
    term3 = (2 / 1) * (x * contraction_p1p1 + 2 * tau_x * contraction_p1p2 + x * contraction_p2p2) * f_prime_s_val
    
    return term1, term2 + term3

def integrate_sigma_hat_prime(tau, C, p1, p2, flavor, Q2):
    def integrand1(x):
        x = tn.tensor(x, device=device)
        tau_x = tau / x
        term1, _ = sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2)
        return term1 * tau_x

    def integrand2(x):
        x = tn.tensor(x, device=device)
        tau_x = tau / x
        _, term2_plus_term3 = sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2)
        return term2_plus_term3 * tau_x

    result1, _ = quad(integrand1, tau, 1)
    result2, _ = quad(integrand2, tau, 1)

    return result1 + result2

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

def process_flavor(args):
    flavor, e_f, g_fR, g_fL, tau, CL, CR, p1, p2, Q2 = args
    integral1 = integrate_sigma_hat_prime(tau, CL, p1, p2, flavor, Q2)
    integral2 = integrate_sigma_hat_prime(tau, CR, p1, p2, flavor, Q2)
    sum_terms_L = summation_terms(Q2, e_f, g_fL)
    sum_terms_R = summation_terms(Q2, e_f, g_fR)

    return (factor * sum_terms_L * integral1, factor * sum_terms_R * integral2)

def d_sigma(Q2, CL, CR, p1, p2, quark_couplings):
    tau = Q2 / s
    d_sigmaL = tn.tensor(0, device=device)
    d_sigmaR = tn.tensor(0, device=device)

    # Prepare arguments for parallel processing
    args = [(flavor, e_f, g_fR, g_fL, tau, CL, CR, p1, p2, Q2) for flavor, e_f, g_fR, g_fL in quark_couplings]

    # Use multiprocessing for parallel processing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_flavor, args)

    for resL, resR in results:
        d_sigmaL += resL
        d_sigmaR += resR

    return 0.38 * 1e9 * (d_sigmaL + d_sigmaR).item()  # Conversion from GeV-2 to Pb and return as scalar

def process_flavor_sm(args):
    flavor, e_f, g_fR, g_fL, tau, Q2 = args
    integral = integrate_sigma_hat_prime_sm(tau, flavor, Q2)
        
    sum_terms_L = summation_terms(Q2, e_f, g_fL)
    sum_terms_R = summation_terms(Q2, e_f, g_fR)
    
    return factor * (sum_terms_L + sum_terms_R) * integral

def integrate_sigma_hat_prime_sm(tau, flavor, Q2):
    def integrand1(x):
        return f_s(x, tau, flavor, Q2) * tau / x
    
    result, _ = quad(integrand1, tau, 1)
    return result

def d_sigma_sm(Q2, quark_couplings):
    tau = Q2 / s
    d_sigma = tn.tensor(0, device=device)

    # Prepare arguments for parallel processing
    args = [(flavor, e_f, g_fR, g_fL, tau, Q2) for flavor, e_f, g_fR, g_fL in quark_couplings]

    # Use multiprocessing for parallel processing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_flavor_sm, args)

    for res in results:
        d_sigma += res

    return 0.38 * 1e9 * d_sigma.item()  # Conversion from GeV^-2 to Pb and return as scalar
    
def sm(Q_min, Q_max, CL, CR, p1, p2, quark_couplings):
    def integ(Q2):
        Q2 = Q2**2
        return d_sigma(Q2, CL, CR, p1, p2, quark_couplings)
    
    return quad(integ, Q_min, Q_max)

if __name__ == '__main__':
    # Initialize CUDA device in the main process
    init_device()

    # Your code to run computations goes here
    # Example: sm(Q_min, Q_max, CL, CR, p1, p2, quark_couplings)
