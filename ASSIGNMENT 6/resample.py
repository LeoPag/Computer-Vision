import numpy as np

def resample(particles,particles_w):

    dim = np.shape(particles)[0]
    indexes = np.arange(0,dim)

    #---NORMALIZING PROBABILITIES---
    sample_indexes = np.random.choice(indexes,size = dim,replace = True, p = particles_w)
    new_particles = particles[sample_indexes, :]
    particles_w = particles_w[sample_indexes]

    return new_particles, particles_w
