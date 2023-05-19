import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  counter = 0
  for i in range(num_corrs):

     index_first_constraint = counter
     index_second_constraint = counter + 1
     x = points2D[i]
     X = points3D[i]
     first_constraint = np.array([0 , 0 , 0 , 0 ,-X[0], -X[1], -X[2], -1, x[1]*X[0], x[1]*X[1], x[1]*X[2], x[1]])

     second_constraint = np.array([X[0],X[1],X[2], 1 , 0 , 0 , 0 , 0 , -x[0] * X[0], -x[0] * X[1], -x[0] * X[2], -x[0]])
     # TODO Add your code here
     constraint_matrix[index_first_constraint] = first_constraint
     constraint_matrix[index_second_constraint] = second_constraint

     counter = counter + 2

  return constraint_matrix
