cmake_minimum_required(VERSION 3.12)

project(ModelOrderReduction VERSION 1.0)

include(CMakeDependentOption)
include(cmake/environment.cmake)

# Specify all required packages
find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.HyperElastic REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
sofa_find_package(Sofa.GL REQUIRED)
sofa_find_package(CollisionOBBCapsule QUIET)
if (NOT CollisionOBBCapsule_FOUND)
    message(WARNING "The package CollisionOBBCapsule has not been found. It is required to compile all the features of ${PROJECT_NAME}")
endif()

# Optional: libtorch for the autoencoder Galerkin path (AEProjector + AEMapping).
# Pass -DTorch_DIR=<libtorch>/share/cmake/Torch on cmake configure.
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Found PyTorch at ${TORCH_INSTALL_PREFIX} — building AE classes")
else()
    message(STATUS "libtorch not found; AEProjector/AEMapping will not be built")
endif()

option(USE_COMPRESSED_ROW_SPARSE "Use compressed row sparse format in the mappings" ON)

set(MOR_SRC "src/ModelOrderReduction")

set(HEADER_FILES
    ${MOR_SRC}/config.h.in

    ${MOR_SRC}/component/forcefield/HyperReducedHelper.h
    ${MOR_SRC}/component/forcefield/HyperReducedHelperKPCA.h
    ${MOR_SRC}/component/forcefield/HyperReducedHexahedronFEMForceField.h
    ${MOR_SRC}/component/forcefield/HyperReducedHexahedronFEMForceField.inl
    ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceField.h
    ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceField.inl
    ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceFieldKPCA.h
    ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceFieldKPCA.inl
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedralCorotationalFEMForceField.h
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedralCorotationalFEMForceField.inl
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceField.h
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceField.inl
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceFieldKPCA.h
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceFieldKPCA.inl
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronHyperelasticityFEMForceField.h
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronHyperelasticityFEMForceField.inl
    ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceField.h
    ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceField.inl
    ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceFieldKPCA.h
    ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceFieldKPCA.inl

    ${MOR_SRC}/component/kernel/KernelProjector.h
    ${MOR_SRC}/component/kernel/PreImageProjector.h

    ${MOR_SRC}/component/loader/MatrixLoader.h
    ${MOR_SRC}/component/loader/MatrixLoader.inl

    ${MOR_SRC}/component/mapping/ModelOrderReductionMapping.h
    ${MOR_SRC}/component/mapping/ModelOrderReductionMapping.inl
    ${MOR_SRC}/component/mapping/KernelPCAMapping.h
    ${MOR_SRC}/component/mapping/KernelPCAMapping.inl
    ${MOR_SRC}/component/mapping/MORContactMapping.h
    ${MOR_SRC}/component/mapping/MORContactMapping.inl

    ${MOR_SRC}/component/contact/MORFrictionContact.h
    ${MOR_SRC}/component/contact/MORFrictionContact.inl
    ${MOR_SRC}/component/contact/MORUnilateralInteractionConstraint.h
    ${MOR_SRC}/component/contact/MORUnilateralInteractionConstraint.inl
    ${MOR_SRC}/component/contact/MORpointModel.h
    ${MOR_SRC}/component/contact/MORpointModel.inl
    )
set(SOURCE_FILES
    ${MOR_SRC}/initModelOrderReduction.cpp

    ${MOR_SRC}/component/forcefield/HyperReducedHelper.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedHelperKPCA.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedHexahedronFEMForceField.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceField.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceFieldKPCA.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedralCorotationalFEMForceField.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceField.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceFieldKPCA.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronHyperelasticityFEMForceField.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceField.cpp
    ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceFieldKPCA.cpp

    ${MOR_SRC}/component/kernel/KernelProjector.cpp
    ${MOR_SRC}/component/kernel/PreImageProjector.cpp

    ${MOR_SRC}/component/loader/MatrixLoader.cpp

    ${MOR_SRC}/component/mapping/ModelOrderReductionMapping.cpp
    ${MOR_SRC}/component/mapping/KernelPCAMapping.cpp
    ${MOR_SRC}/component/mapping/MORContactMapping.cpp

    ${MOR_SRC}/component/contact/MORFrictionContact.cpp
    ${MOR_SRC}/component/contact/MORUnilateralInteractionConstraint.cpp
    ${MOR_SRC}/component/contact/MORpointModel.cpp
    )
set(README_FILES
    README.md
    )


if (CollisionOBBCapsule_FOUND)
    list(APPEND SOURCE_FILES ${MOR_SRC}/component/contact/MORFrictionContact[OBBCapsule].cpp)
endif()

if(Torch_FOUND)
    list(APPEND HEADER_FILES
        ${MOR_SRC}/component/kernel/AEProjector.h
        ${MOR_SRC}/component/mapping/AEMapping.h
        ${MOR_SRC}/component/mapping/AEMapping.inl
        ${MOR_SRC}/component/forcefield/HyperReducedHelperAE.h
        ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceFieldAE.h
        ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceFieldAE.inl
        ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceFieldAE.h
        ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceFieldAE.inl
        ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceFieldAE.h
        ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceFieldAE.inl
    )
    list(APPEND SOURCE_FILES
        ${MOR_SRC}/component/kernel/AEProjector.cpp
        ${MOR_SRC}/component/mapping/AEMapping.cpp
        ${MOR_SRC}/component/forcefield/HyperReducedHelperAE.cpp
        ${MOR_SRC}/component/forcefield/HyperReducedTetrahedronFEMForceFieldAE.cpp
        ${MOR_SRC}/component/forcefield/HyperReducedTriangleFEMForceFieldAE.cpp
        ${MOR_SRC}/component/forcefield/HyperReducedRestShapeSpringsForceFieldAE.cpp
    )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MODELORDERREDUCTION")
target_link_libraries(${PROJECT_NAME}
    Sofa.Core
    Sofa.Component.Collision.Response.Contact
    Sofa.Component.SolidMechanics.FEM.HyperElastic
    Sofa.Component.SolidMechanics.FEM.Elastic
    Sofa.GL
)

if (CollisionOBBCapsule_FOUND)
    target_link_libraries(${PROJECT_NAME} CollisionOBBCapsule)
endif()

if(Torch_FOUND)
    target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC SOFA_BUILD_AUTOENCODER)
endif()

# Standalone parity-test dumper for the kPCA KernelProjector.
add_executable(KernelProjector_dump tools/KernelProjector_dump.cpp)
target_link_libraries(KernelProjector_dump ${PROJECT_NAME})

# Standalone parity-test dumper for the explicit pre-image head.
add_executable(PreImageProjector_dump tools/PreImageProjector_dump.cpp)
target_link_libraries(PreImageProjector_dump ${PROJECT_NAME})

if(Torch_FOUND)
    add_executable(AEProjector_dump tools/AEProjector_dump.cpp)
    target_link_libraries(AEProjector_dump ${PROJECT_NAME} ${TORCH_LIBRARIES})
endif()

# Standalone parity-test dumper for HyperReducedHelperKPCA.
add_executable(HyperReducedHelperKPCA_dump tools/HyperReducedHelperKPCA_dump.cpp)
target_link_libraries(HyperReducedHelperKPCA_dump ${PROJECT_NAME})

# Config files and install rules for pythons scripts
find_file(SofaPython3Tools NAMES "SofaPython3/lib/cmake/SofaPython3/SofaPython3Tools.cmake")
if(SofaPython3Tools)
    message("-- Found SofaPython3Tools.")
    include(${SofaPython3Tools})
else()
    # try again with the find_package mechanism
    find_package(SofaPython3 QUIET)
endif()
if(SofaPython3Tools OR SofaPython3_FOUND)
    if(NOT SP3_PYTHON_PACKAGES_DIRECTORY)
        set(SP3_PYTHON_PACKAGES_DIRECTORY "python3/site-packages")
    endif()
    message("-- Python3 packages will be installed in: ${SP3_PYTHON_PACKAGES_DIRECTORY}.")
    SP3_add_python_package(
            SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/mor
            TARGET_DIRECTORY mor
    )
    SP3_add_python_package(
            SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/morlib
            TARGET_DIRECTORY morlib
    )
endif()

# Move all resources when install: examples, doc and python files
install(FILES "README.md" "LICENSE" DESTINATION .)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doc")
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/doc/ModelOrderReduction.pdf DESTINATION doc COMPONENT resources)	
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doc/examples")
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples DESTINATION . COMPONENT resources)
    endif()
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools DESTINATION . COMPONENT resources)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ModelOrderReduction
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "ModelOrderReduction"
    RELOCATABLE "plugins"
    )

# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(MODELORDERREDUCTION_BUILD_TESTS "Compile the tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(MODELORDERREDUCTION_BUILD_TESTS)
    add_subdirectory(ModelOrderReduction_test)
endif()

include(cmake/packaging.cmake)
