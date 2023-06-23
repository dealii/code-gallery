INCLUDE_DIRECTORIES(${SRC_PATH})
SET(TEST_LIBRARIES ${LIB_NAME})

IF(DO_MFG)
  DEAL_II_PICKUP_TESTS()
ELSE()
  MESSAGE(STATUS "Testing disabled: Manufactured solutions option not enabled.")
ENDIF()
