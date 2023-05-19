import numpy as np

def propagate(particles, frame_height, frame_width,params):
    particles = np.transpose(particles)
    deltat = 1

    if params['model'] == 0:
        A_matrix = [[1,0],[0,1]]
        position_noise =params['sigma_position']
        noise_deviations = [position_noise,position_noise]

    elif params['model'] == 1:
        A_matrix = [[1,0,deltat,0],[0,1,0,deltat],[0,0,1,0],[0,0,0,1]]
        position_noise = params['sigma_position']
        velocity_noise = params['sigma_velocity']
        noise_deviations = [position_noise, position_noise, velocity_noise, velocity_noise]

    for i in range(particles.shape[1]):
        random_sample = np.random.normal(0,1,particles.shape[0])
        noise = noise_deviations * random_sample
        particles[:, i] = np.matmul(A_matrix,particles[:,i]) + np.transpose(noise)

        #---CHECKING THE BOUNDARIES---
        if(particles[0,i] < 0):
            particles[0,i] = 0

        if(particles[1,i] < 0):
            particles[1:i] = 0

        if(particles[0,i]) > frame_width:
            particles[0,i] = frame_width

        if(particles[1,i]) > frame_height:
            particles[1,i] = frame_height

    particles = np.transpose(particles)

    return(particles)
