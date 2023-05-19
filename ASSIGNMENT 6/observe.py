import numpy as np
import math

from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width,hist_bin, hist, sigma_observe):

    particles_w = np.array([])
    for i in range(particles.shape[0]):
        x_center = particles[i,0]
        y_center = particles[i,1]

        #___THIS IS TO MAKE SURE ALL THE BOXES HAVE THE SAME DIMENSIONS___
        x_center = max(x_center, bbox_width / 2)
        x_center = min(x_center, frame.shape[1] - bbox_width / 2)
        y_center = max(y_center, bbox_height / 2)
        y_center = min(y_center, frame.shape[0] - bbox_height / 2)

        xmin = x_center - (bbox_width / 2)
        xmax = x_center + (bbox_width / 2)
        ymin = y_center - (bbox_height / 2)
        ymax = y_center + (bbox_height / 2)


        current_hist = color_histogram(xmin,ymin,xmax,ymax,frame,hist_bin)
        chi_quadro_dist = chi2_cost(current_hist,hist)
        pi = (np.exp(-(chi_quadro_dist**2)/ (2 * (sigma_observe**2)))) / (sigma_observe * np.sqrt(2*math.pi))

        particles_w = np.append(particles_w,pi)
        particles_w = particles_w / np.sum(particles_w)

    return(particles_w)
