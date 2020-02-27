#include <gtest/gtest.h>
#include "autorally_control/path_integral/vector_math.cuh"

#include <math.h>

TEST(VectorMathTest, multiply_vector3_by_scalar){
  float v[3] = {-1.6, 2.2, 3};
  float result[3] = {0, 0, 0};
  multiplyVector3ByScalar(v, 2.5, result);

  EXPECT_DOUBLE_EQ(-4.0, result[0]);
  EXPECT_DOUBLE_EQ(5.5, result[1]);
  EXPECT_DOUBLE_EQ(7.5, result[2]);
}

TEST(VectorMathTest, test_vector3_addition){
  float v1[3] = {1, 2, 3};
  float v2[3] = {4, -5.5, 6};
  float result[3] = {0, 0, 0};
  addVector3(v1, v2, result);

  EXPECT_DOUBLE_EQ(5, result[0]);
  EXPECT_DOUBLE_EQ(-3.5, result[1]);
  EXPECT_DOUBLE_EQ(9, result[2]);
}

TEST(VectorMathTest, test_matrix3x3_vector3_multiplication){
  float v[3] = {1, 2, 3};
  float M[3][3] = {
    {4, -2, 5.5},
    {1, 2, 3},
    {3.7, 2.33, 4.1},
  };
  float result[3] = {0, 0, 0};
  multiplyVector3By3x3Matrix(M, v, result);

  EXPECT_NEAR(16.5, result[0], 1e-5);
  EXPECT_NEAR(14.0, result[1], 1e-5);
  EXPECT_NEAR(20.66, result[2], 1e-5);
}

TEST(VectorMathTest, test_dot_product_vector3){
  float v1[3] = {1, 2, 3};
  float v2[3] = {3.4, 5.3, 29};
  
  EXPECT_DOUBLE_EQ(101, dotProductVector3(v1, v2));
}

TEST(VectorMathTest, test_cross_product_vector3){
  float v1[3] = {1, 2, 3};
  float v2[3] = {3.4, 5.3, 29};
  float result[3] = {0, 0, 0};
  crossProductVector3(v1, v2, result);
  
  EXPECT_NEAR(42.1, result[0], 1e-5);
  EXPECT_NEAR(-18.8, result[1], 1e-5);
  EXPECT_NEAR(-1.5, result[2], 1e-5);
}

TEST(VectorMathTest, rotate_vector3_about_z_axis_90_deg_counterclockwise){
  float v[3] = {2.99, 3.45, 99};
  float result[3] = {0, 0, 0};
  rotateVector3AboutZAxis(v, M_PI/2, result);
  
  EXPECT_NEAR(-3.45, result[0], 1e-5);
  EXPECT_NEAR(2.99, result[1], 1e-5);
  EXPECT_NEAR(99, result[2], 1e-5);
}

TEST(VectorMathTest, rotate_vector3_about_z_axis_90_deg_clockwise){
  float v[3] = {2.99, 3.45, 99};
  float result[3] = {0, 0, 0};
  rotateVector3AboutZAxis(v, -M_PI/2, result);
  
  EXPECT_NEAR(3.45, result[0], 1e-5);
  EXPECT_NEAR(-2.99, result[1], 1e-5);
  EXPECT_NEAR(99, result[2], 1e-5);
}

TEST(VectorMathTest, rotate_vector3_about_z_axis_27_deg_clockwise){
  float v[3] = {2.99, 3.45, 99};
  float result[3] = {0, 0, 0};
  rotateVector3AboutZAxis(v, -27.0 * M_PI/180, result);
  
  EXPECT_NEAR(4.2304, result[0], 1e-4);
  EXPECT_NEAR(1.7165, result[1], 1e-4);
  EXPECT_NEAR(99, result[2], 1e-4);
}

TEST(VectorMathTest, create_rotation_matrix_about_z_axis){
  float rot_rad = 482.13432;
  float result[3][3] = {
    {0,0,0}, 
    {0,0,0}, 
    {0,0,0},
  };
  createRotationMatrixAboutZAxis(rot_rad, result);

  EXPECT_NEAR(-0.09998, result[0][0], 1e-4);
  EXPECT_NEAR( 0.99499, result[0][1], 1e-4);
  EXPECT_NEAR( 0.00000, result[0][2], 1e-4);

  EXPECT_NEAR(-0.99499, result[1][0], 1e-4);
  EXPECT_NEAR(-0.09998, result[1][1], 1e-4);
  EXPECT_NEAR( 0.00000, result[1][2], 1e-4);

  EXPECT_NEAR( 0.00000, result[2][0], 1e-4);
  EXPECT_NEAR( 0.00000, result[2][1], 1e-4);
  EXPECT_NEAR( 1.00000, result[2][2], 1e-4);
}

TEST(VectorMathTest, get_unit_vector_in_direction){
  float v[3] = {2.99, 3.45, 99};
  float result[3] = {0, 0, 0};
  getUnitVectorInDirection(v, result);
  
  EXPECT_NEAR(0.030170, result[0], 1e-5);
  EXPECT_NEAR(0.034811, result[1], 1e-5);
  EXPECT_NEAR(0.998938, result[2], 1e-5);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}
