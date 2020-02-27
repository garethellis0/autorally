#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H
/**
 * @brief A small vector math library, intended for use with CUDA code
 */

#include <math.h>

// TODO: handle cases where the output is the same vector as (one of) the input(s)
//       *PARTICUARLY ADDITION*

// This macro allows us to compile functions without nvcc for unit testing
// purposes
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

/**
 * Multiply the given vector by a scalar
 * 
 * @param[in] v The vector
 * @param[in] s The scalar
 * @param[out] result This will be set to s*v
 */
CUDA_HOSTDEV static void multiplyVector3ByScalar(const float v[3], const float s, float result[3]){
  for (int i = 0; i < 3; i++){
    result[i] = v[i]*s;
  }
}

/**
 * Add two vectors of length 3 together
 * 
 * @param[in] v1 The first vector
 * @param[in] v2 The second vector
 * @param[out] result This will be set to v1+v2. 
 */
CUDA_HOSTDEV static void addVector3(const float v1[3], const float v2[3], float result[3]){
  for (int i = 0; i < 3; i++){
    result[i] = v1[i] + v2[i];
  }
}

/**
 * Computes the result of multiplying a given vector by a given matrix
 * 
 * ie. return M*v
 * 
 * @param[in] M The matrix to multiply the given vector by.
 * @param[in] v The vector to multiply by the given matrix.
 * @param[out] result This will be set to the result of M*v
 */
CUDA_HOSTDEV static void multiplyVector3By3x3Matrix(const float M[3][3], const float v[3], float result[3]){
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
 * @param[in] v1 The first vector
 * @param[in] v2 The second vector
 * 
 * @return The dot product of v1 and v2
 */
CUDA_HOSTDEV static float dotProductVector3(const float v1[3], const float v2[3]){
  float result = 0;
  for (int i = 0; i < 3; i++){
    result += v1[i]*v2[i];
  }
  return result;
}

/**
 * Compute the cross product of two vectors
 * 
 * @param[in] v1 The first vector
 * @param[in] v2 The second vector
 * @param[out] result The result of taking the cross product of v1 and v2
 */
CUDA_HOSTDEV static void crossProductVector3(const float v1[3], const float v2[3], float result[3]){
  result[0] = v1[1]*v2[2] - v1[2]*v2[1];
  result[1] = v1[2]*v2[0] - v1[0]*v2[2];
  result[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

/**
 * Constructs a matrix that can be multipled by a vector to rotate it
 *
 * Rotation is *counterclockwise*, and about the z-axis
 * 
 * @param[in] rotation_rad The amount to rotate the vector by, in radians
 * @param[out] result The created rotation matrix
 */
CUDA_HOSTDEV static void createRotationMatrixAboutZAxis(const float rotation_rad, float result[3][3]){
  result[0][0] = cos(rotation_rad);
  result[0][1] = -sin(rotation_rad);
  result[0][2] = 0.0;
  result[1][0] = sin(rotation_rad);
  result[1][1] = cos(rotation_rad);
  result[1][2] = 0.0;
  result[2][0] = 0.0;
  result[2][1] = 0.0;
  result[2][2] = 1.0;
}

/**
 * Rotate the given vector by the given amount 
 *
 * Rotation is *counterclockwise*, and about the z-axis
 * 
 * @param[in] v The vector to rotate
 * @param[in] rotation_rad The amount to rotate the vector by, in radians
 * @param[out] result This vector will be set to the result of rotating the
 *                    given vector by the given amount in the 
 *                    *counterclockwise* direcition
 */
CUDA_HOSTDEV static void rotateVector3AboutZAxis(const float v[3], const float rotation_rad, float result[3]){
  float rotation_matrix[3][3];
  createRotationMatrixAboutZAxis(rotation_rad, rotation_matrix);

  multiplyVector3By3x3Matrix(rotation_matrix, v, result);
}

/**
 * Gets the unit vector in the direction of the given vector
 *
 * @param[in] dir The direction to get a unit vector in
 * @param[out] result A unit vector in the direction of `dir`
 */
CUDA_HOSTDEV static void getUnitVectorInDirection(const float dir[3], float result[3]){
  float length = 0;
  for (int i = 0; i < 3; i++){
    length += pow(dir[i], 2.0);
  }
  length = sqrt(length);

  for (int i = 0; i < 3; i++){
    result[i] = dir[i] / length;
  }
}

#endif // VECTOR_MATH_H
