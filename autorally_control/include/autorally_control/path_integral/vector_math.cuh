#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

// This macro allows us to compile functions without nvcc for unit testing
// purposes
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

// TODO: if we can specify the size as part of the arg, should remove\
//       that requirement from all the jdocs, it's duplicate info

/**
 * Add two vectors of length 3 together
 * 
 * @param[in] v1 The first vector, assumed to be a 3-long array
 * @param[in] v2 The second vector, assumed to be a 3-long array
 * @param[out] result This will be set to v1+v2. 
 *                    This must be pre-allocated to be of at least length 3.
 */
CUDA_HOSTDEV static void addVector3(float v1[3], float v2[3], float result[3]){
  for (int i = 0; i < 3; i++){
    result[i] = v1[i] + v2[i];
  }
}

/**
 * Computes the result of multiplying a given vector by a given matrix
 * 
 * ie. return M*v
 * 
 * @param[in] M The matrix to multiply the given vector by
 * @param[in] v The vector to multiply by the given matrix
 * @param[out] result This will be set to the result of M*v, it must be 
 *                    pre-allocated to be of at least length 3.
 */
CUDA_HOSTDEV static void multiplyVector3By3x3Matrix(float M[3][3], float v[3], float result[3]){
  for (int i = 0; i < 3; i++){
    result[i] = 0;
    for (int j = 0; j < 3; j++){
      result[i] += M[i][j]*v[j];
    }
  }
}


/**
 * Compute the dot product of two vectors
 * 
 * @param[in] v1 The first vector, assumed to be a 3-long array
 * @param[in] v2 The second vector, assumed to be a 3-long array
 * 
 * @return The dot product of v1 and v2
 */
CUDA_HOSTDEV static float dotProductVector3(float v1[3], float v2[3]){
  float result = 0;
  for (int i = 0; i < 3; i++){
    result += v1[i]*v2[i];
  }
  return result;
}

/**
 * Compute the cross product of two vectors
 * 
 * @param[in] v1 The first vector, assumed to be a 3-long array
 * @param[in] v2 The second vector, assumed to be a 3-long array
 * @param[out] result The result of taking the cross product of v1 and v2. 
 *                    This must be pre-allocated to be of at least length 3.
 */
CUDA_HOSTDEV static void crossProductVector3(float v1[3], float v2[3], float result[3]){
  result[0] = v1[1]*v2[2] - v1[2]*v2[1];
  result[1] = v1[2]*v2[0] - v1[0]*v2[2];
  result[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


#endif // VECTOR_MATH_H
