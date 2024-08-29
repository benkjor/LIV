import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch as tn
import random
import warnings
import multiprocessing as mp

from scipy.integrate import quad, IntegrationWarning
import time

# Import files
from constants import *
from pdfpy import *
from functions import  d_sigma, d_sigma_sm
from rotation import *
# Quarks
quarks = [
    (2, 2/3*e, 'u', 1/2),
     (1, -1/3*e, 'd', -1/2),
     (3, -1/3*e, 's', -1/2),
     (4, 2/3*e, 'c', 1/2),
      (5, -1/3*e, 'b', -1/2),
    #  (6, 2/3*e, 't', 1/2),
]

# List of quark properties and couplings
quark_couplings = []

for flavor, e_f, name, I3 in quarks:
    g_fR = -e_f * sin2th_w
    g_fL = I3 - e_f * sin2th_w
    
    # Rounding to 4 decimal places
    e_f = round(e_f, 4)
    g_fR = round(g_fR, 4)
    g_fL = round(g_fL, 4)
    
    quark_couplings.append((flavor, e_f, g_fR, g_fL))

print(quark_couplings)

#Don't foregt the metric convenction (+, -, -, -)
g = tn.tensor([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,-1]
], dtype=tn.float32)
CL1 = tn.tensor([
    [0, 0, 0, 0],
    [0, 1e-5, 0, 0],
    [0, 0, -1e-5, 0],
    [0,0, 0, 0]
], dtype=tn.float32)
CL2 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, -1e-5, 0],
    [0, -1e-5, 0, 0],
    [0,0, 0, 0]
], dtype=tn.float32)
CL3 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, -1e-5],
    [0, 0, 0, 0],
    [0,-1e-5, 0, 0]
], dtype=tn.float32)
CL4 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, -1e-5],
    [0,0,-1e-5, 0]
], dtype=tn.float32)


CR = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=tn.float32)

# C0 = tn.tensor([
#     [0, 0, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0]
# ], dtype=tn.float32)

# Define the constant tensors once
A = 1 / np.sqrt(2)
p1 = A * tn.tensor([1, 0, 0, 1], dtype=tn.float32)
p2 = A * tn.tensor([1, 0, 0, -1], dtype=tn.float32)

# Precompute total number of steps

specific_time = datetime(2018, 1, 1, 0, 0)

start_time = int(specific_time.timestamp())

# start_time = int(time.time())
end_time = start_time + int(timedelta(days=1).total_seconds())
step_seconds = int(timedelta(hours=0.25).total_seconds())
num_steps = (end_time - start_time) // step_seconds

# Lists to store the times and contr matrix elements
times = []
contrelep1 = []
contrelep2 = []

R_y_lat = R_y(latitude)
R_x_azi = R_x(azimuth)

# Main loop
current_time = start_time
for _ in range(num_steps):
    # Convert current_time to a timestamp
    current_datetime = datetime.fromtimestamp(current_time)
    time_utc = current_datetime.timestamp()

    # Calculate omega_t
    omega_t_sid = omega_utc * time_utc + 3.2830 
    # Construct the complete rotation matrix from SCF to CMS
    R_Z_omega = R_Z(omega_t_sid)
    
    R_mat = tn.matmul(R_y_lat, tn.matmul(R_x_azi, tn.matmul(R_z, R_Z_omega)))
    R_matrix1 = tn.einsum('am,na->mn', g, R_mat)
    R_matrix2 = tn.einsum('ma,an->mn', g, R_mat)

    # Compute contrL and contrR using matrix multiplication
    contrp1 = tn.einsum('ij,j->i', R_matrix1, p1)
    contrp2 =  tn.einsum('ij,i->j',R_matrix2, p2)

    # Record the times and contr matrix elements
    times.append(current_time)
    contrelep1.append(contrp1)
    contrelep2.append(contrp2)


    # Move to the next time step
    current_time += step_seconds
    
# Generate a list random of Q values 
Q_values = []
for _ in range(len(contrelep1)):
    while True:
        Q = np.random.normal(loc=75, scale=2)
        if 70 <= Q <= 80:
            Q_values.append(Q)
            break
            
warnings.simplefilter("ignore", IntegrationWarning)

def compute_result(args):
    pm, pn, Q, quark_couplings, CL1, CL2, CL3, CL4, CR = args
    
    # Calculate Q^2
    Q2 = Q**2
    
    # Perform all calculations with the same Q for this iteration
    result_sm = d_sigma_sm(Q2, quark_couplings)
    result_sme1 = d_sigma(Q2, CL1, CR, pm, pn, quark_couplings)
    result_sme2 = d_sigma(Q2, CL2, CR, pm, pn, quark_couplings)
    result_sme3 = d_sigma(Q2, CL3, CR, pm, pn, quark_couplings)
    result_sme4 = d_sigma(Q2, CL4, CR, pm, pn, quark_couplings)
    
    # Return the result as a dictionary
    return {
        'Q': Q,
        'result_sm': result_sm,
        'result_sme1': result_sme1,
        'result_sme2': result_sme2,
        'result_sme3': result_sme3,
        'result_sme4': result_sme4
    }

# Prepare the arguments for parallel processing
args_list = [(pm, pn, Q, quark_couplings, CL1, CL2, CL3, CL4, CR) 
             for (pm, pn), Q in zip(zip(contrelep1, contrelep2), Q_values)]

# Create a multiprocessing Pool
with mp.Pool(mp.cpu_count()) as pool:
    # Map the compute_result function to the args_list
    results = pool.map(compute_result, args_list)

# Function to convert timestamps to hours
def convto_hours(timestamps):
    start_time = timestamps[0]  # The start time to normalize
    return [(t - start_time) / 3600 for t in timestamps]  # Convert seconds to hours

# Perform conversion
hours_start = convto_hours(times)

dratio1l = [result['result_sme1'] / result['result_sm'] for result in results]
dratio2l = [result['result_sme2'] / result['result_sm'] for result in results]
dratio3l = [result['result_sme3'] / result['result_sm'] for result in results]
dratio4l = [result['result_sme4'] / result['result_sm'] for result in results]

dratio1 = np.array(dratio1l)
dratio2 = np.array(dratio2l)
dratio3 = np.array(dratio3l)
dratio4 = np.array(dratio4l)
hours_array = np.array(hours_start)

plt.figure(figsize=(12, 6))

# Plot data
plt.step(hours_array, dratio1, where='post', color='r', label='$C_{L,XX} = - C_{L,YY} = 10^{-5}$', linewidth=1.5)
plt.step(hours_array, dratio2, where='post', color='b', label='$C_{L,XY} = C_{L,YX} = 10^{-5}$', linewidth=1.5)
plt.step(hours_array, dratio3, where='post', color='g', label='$C_{L,XZ} = C_{L,ZX} = 10^{-5}$', linewidth=1.5)
plt.step(hours_array, dratio4, where='post', color='gold', label='$C_{L,YZ} = C_{L,ZY} = 10^{-5}$', linewidth=1.5)

# Customizing the legend
plt.legend(loc='upper left', bbox_to_anchor=(0.9, 1), fontsize=10, frameon=True)

# Adding labels and title 
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('$d\\sigma_{SME}/d\\sigma_{SM}$', fontsize=12)
plt.title('$SME/SM \; at \; Q \in [70,80] \;GeV$', fontsize=16, loc='left')
plt.text(12, 1.008, '2018', fontsize=20, horizontalalignment='center')

plt.minorticks_on()
plt.tick_params(axis='x', which='minor', bottom=False)  
plt.tick_params(which='both', width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='gray')
plt.tick_params(axis='y', direction='in', which ='both') 

plt.xticks(ticks=range(0, 25, 1), labels=[str(hour) for hour in range(0, 25, 1)])

plt.tight_layout(rect=[0, 0, 0.8, 0.8]) 

# Showing the plot
plt.savefig("liv.png", bbox_inches='tight', pad_inches=0.1)
plt.show()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plot data in each subplot
axs[0, 0].step(hours_array, dratio1, where='post', color='r', label='$C_{L,XX} = - C_{L,YY} = 10^{-5}$')
axs[0, 0].set_title('Data 1')
axs[0, 0].set_xlabel('Time (hours)')
axs[0, 0].set_ylabel('$ratios$')
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].step(hours_array, dratio2, where='post', color='b', label='$C_{L,XY} = C_{L,YX} = 10^{-5}$')
axs[0, 1].set_title('Data 2')
axs[0, 1].set_xlabel('Time (hours)')
axs[0, 1].set_ylabel('$ratios$')
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].step(hours_array, dratio3, where='post', color='g', label='$C_{L,XZ} = C_{L,ZX} = 10^{-5}$')
axs[1, 0].set_title('Data 3')
axs[1, 0].set_xlabel('Time (hours)')
axs[1, 0].set_ylabel('$ratios$')
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].step(hours_array, dratio4, where='post', color='black', label='$C_{L,YZ} = C_{L,ZY} = 10^{-5}$')
axs[1, 1].set_title('Data 4')
axs[1, 1].set_xlabel('Time (hours)')
axs[1, 1].set_ylabel('$ratios$')
axs[1, 1].grid(True)
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.savefig("plots.png")
# Show the plot
plt.show()

resultssm = [d_sigma_sm(75**2,quark_couplings)]
# Assume resultssm is a list with the result from the SM calculation

# Generate sinusoidal pseudo-data over 24 time steps with a 0.01 deviation
time_steps = np.arange(24)
amplitude = 0.01 * resultssm[0]  # 1% of the SM result as the amplitude of the sinusoid
np.random.seed(42)  # For reproducibility
pseudo_data = resultssm[0] + amplitude * np.sin(2 * np.pi * time_steps / 24+np.pi*8/24) 

# Plot the pseudo-data
plt.figure(figsize=(10, 6))
plt.plot(time_steps, pseudo_data, marker='o', linestyle='-', label='Pseudo-data')
plt.axhline(y=resultssm[0], color='r', linestyle='--', label='SM Result')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Sinusoidal Pseudo-data Over 24 Time Steps')
plt.legend()
plt.grid(True)
plt.savefig("pseudo.png")
plt.show()

# Create the linear values
linear_values = tn.linspace(-0.025, 0.025, steps=20)

# Initialize an empty list to hold the tensors
C_values = []

# Loop through the linear values and create the tensors
for value in linear_values:
    tensor = tn.zeros((4,4))  # Create a tensor of the specified size, filled with zeros
    tensor[(2,3)] = value  # Set the value at the first index
    tensor[(3,2)] = value  # Set the value at the second index
    C_values.append(tensor)  # Append the tensor to the list


def d_sigma_variation(args):
    C, Q2, pm, pn, quark_couplings, CR = args
    return d_sigma(Q2, C, CR, pm, pn, quark_couplings)

# Prepare the arguments for parallel processing
args_list = [
    (C, Q**2, pm, pn, quark_couplings, CR)
    for C in C_values
    for (pm, pn), Q in zip(zip(contrelep1, contrelep2), Q_values)
]

# Create a multiprocessing Pool
with mp.Pool(mp.cpu_count()) as pool:
    # Map the calculate_d_sigma_parallel function to the args_list
    flat_results = pool.map(d_sigma_variation, args_list)

# Reshape the flat_results back into a nested list
d_sigma_results = [
    flat_results[i * len(Q_values):(i + 1) * len(Q_values)]
    for i in range(len(C_values))
]
# Plotting each d_sigma result for each C_value
for i, C in enumerate(C_values):
    plt.plot(range(len(d_sigma_results[i])), d_sigma_results[i], marker='o', label=f'C = {C}')

plt.xlabel('Index of pm, pn pair')
plt.ylabel('d_sigma')
plt.title('d_sigma results for different C values')
plt.grid(True)
plt.savefig("variation.png")
plt.show()

chi2_values = []
for results in d_sigma_results:
    chi2 = np.sum([(obs - res)**2 / obs for obs, res in zip(pseudo_data, results) if obs != 0])
    chi2_values.append(chi2)

# Plotting the chi-square as a function of C
plt.plot(linear_values, chi2_values, marker='o')
plt.xlabel('C')
plt.ylabel('Chi-Square')
plt.title('$\chi^2$ as a function of C')

ax = plt.gca()
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.grid(True)
plt.savefig("chi2.png")
plt.show()

