// tests of SWIG interface code
// Author: Karl Wette, 2011

// Not all of the static vector/matrix member/global variable/enum
// combinations appear to exist in LAL code, so the following definitions
// exist to ensure all possible combinations are tested. They also test
// the SWIGLAL*_ELEM macros, which provides *_getel and *_setel methods.
%inline %{
  enum swiglal_static_test_enum {
    swiglal_static_test_enum_a
  };
  struct swiglal_static_test_struct {
    int vector[3];
    int matrix[2][3];
    swiglal_static_test_enum enum_vector[3];
    swiglal_static_test_enum enum_matrix[2][3];
  };
  int swiglal_static_test_vector[3];
  int swiglal_static_test_matrix[2][3];
  swiglal_static_test_enum swiglal_static_test_enum_vector[3];
  swiglal_static_test_enum swiglal_static_test_enum_matrix[2][3];
  const int swiglal_static_test_const_vector[3] = {1, 2, 4};
%}
SWIGLAL_FIXED_1DARRAY_ELEM(int, vector,
                             swiglal_static_test_struct);
SWIGLAL_FIXED_2DARRAY_ELEM(int, matrix,
                             swiglal_static_test_struct);
SWIGLAL_FIXED_1DARRAY_ELEM(swiglal_static_test_enum, enum_vector,
                             swiglal_static_test_struct);
SWIGLAL_FIXED_2DARRAY_ELEM(swiglal_static_test_enum, enum_matrix,
                             swiglal_static_test_struct);
SWIGLAL_GLOBAL_FIXED_1DARRAY_ELEM(int, swiglal_static_test_vector);
SWIGLAL_GLOBAL_FIXED_2DARRAY_ELEM(int, swiglal_static_test_matrix);
SWIGLAL_GLOBAL_FIXED_1DARRAY_ELEM(swiglal_static_test_enum, swiglal_static_test_enum_vector);
SWIGLAL_GLOBAL_FIXED_2DARRAY_ELEM(swiglal_static_test_enum, swiglal_static_test_enum_matrix);
SWIGLAL_GLOBAL_CONST_FIXED_1DARRAY_ELEM(int, swiglal_static_test_const_vector);
