include_directories(${SRC_PATH})
set(TEST_LIBRARIES ${LIB_NAME} ${BRAID_DIR}/libbraid.a)
DEAL_II_PICKUP_TESTS()
