import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, HeliocentricTrueEcliptic, ITRS, EarthLocation
import torch as tn
from time import sleep
from datetime import datetime, timedelta, timezone
from math import pi

tn.set_printoptions(precision=10, sci_mode=False)

# Define the location of CMS in terms of longitude, latitude and azimuth

azimuth = 1.7677    
latitude = 0.8082  
longitude = 0.1061   

# Define the Earth's angular velocity (rad/s)
omega_utc = 2*pi/(86164)     # Earth's angular velocity in rad/s at UTC.
omega_siderial = 2*pi/(86400)
# Rotation matrices to go from ITRS to CMS frame

# rotation around the z-axis which makes the x-axis normal to the plane of the LHC.

R_z = tn.tensor([
        [1,0,0,0],
        [0,0,-1,0],
        [0,1,0,0],
        [0,0,0,1]
    ], dtype=tn.float32)


# To orient the z-axis towards the North. We rotate counterclockwise around the x′ axis with an angle π −θ (co-azimuth).
def R_x(angle):
    return tn.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, - np.cos(angle), np.sin(angle)],
        [0, 0, - np.sin(angle), - np.cos(angle)]
    ], dtype=tn.float32)


# Rotation around the y axis to align the z-axis with the Z-axis of the SCF.
def R_y(angle):
    return tn.tensor([
        [1, 0, 0, 0],
        [0, np.cos(angle), 0, np.sin(angle)],
        [0, 0, 1, 0],
        [0, -np.sin(angle), 0, np.cos(angle)]
    ], dtype=tn.float32)


# A final rotation around the Z-axis has two purposes: to follow the rotation of the Earth over time and to synchronize with the SCF:
def R_Z(angle):
    return tn.tensor([
        [1, 0 , 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ], dtype=tn.float32)


