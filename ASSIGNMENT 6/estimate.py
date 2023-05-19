import numpy as np

def estimate(particles, particles_w):

    return(np.average(particles,0,particles_w.flatten()))
