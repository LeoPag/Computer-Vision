import numpy as np
import math

def color_histogram(xmin,ymin,xmax,ymax,frame,hist_bin):
    xmin = max(1,round(xmin))
    xmax = min(round(xmax),160)
    ymin = max(1,round(ymin))
    ymax = min(round(ymax),120)
    red = frame[ymin:ymax,xmin:xmax,0]
    green = frame[ymin:ymax,xmin:xmax,1]
    blue = frame[ymin:ymax,xmin:xmax,2]
    red_hist,_ = np.histogram(red, hist_bin, (0,255))
    green_hist,_ = np.histogram(green, hist_bin, (0,255))
    blue_hist,_ = np.histogram(blue, hist_bin, (0,255))
    histogram = np.hstack((red_hist,green_hist,blue_hist))
    sum_histo = np.sum(histogram)
    histogram_norm = histogram / sum_histo
    return(histogram_norm)
