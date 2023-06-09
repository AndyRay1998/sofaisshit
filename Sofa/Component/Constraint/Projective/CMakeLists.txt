cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Constraint.Projective LANGUAGES CXX)

set(SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR "src/sofa/component/constraint/projective")

set(HEADER_FILES
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/AffineMovementConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/AffineMovementConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/AttachConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/AttachConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedPlaneConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedPlaneConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedRotationConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedRotationConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedTranslationConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedTranslationConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/HermiteSplineConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/HermiteSplineConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/LinearMovementConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/LinearMovementConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/LinearVelocityConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/LinearVelocityConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/OscillatorConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/OscillatorConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ParabolicConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ParabolicConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PartialFixedConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PartialFixedConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PartialLinearMovementConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PartialLinearMovementConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PatchTestMovementConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PatchTestMovementConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PositionBasedDynamicsConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PositionBasedDynamicsConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/SkeletalMotionConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/SkeletalMotionConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToLineConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToLineConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToPlaneConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToPlaneConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToPointConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToPointConstraint.inl
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectDirectionConstraint.h
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectDirectionConstraint.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/AffineMovementConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/AttachConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedPlaneConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedRotationConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/FixedTranslationConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/HermiteSplineConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/LinearMovementConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/LinearVelocityConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/OscillatorConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ParabolicConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PartialFixedConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PartialLinearMovementConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PatchTestMovementConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/PositionBasedDynamicsConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/SkeletalMotionConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToLineConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToPlaneConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectToPointConstraint.cpp
    ${SOFACOMPONENTCONSTRAINTPROJECTIVE_SOURCE_DIR}/ProjectDirectionConstraint.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
