/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONFEMFORCEFIELDRESIDUALKERNEL_CPP
#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronFEMForceFieldResidualKernel.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <ModelOrderReduction/config.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;


void registerHyperReducedTetrahedronFEMForceFieldResidualKernel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Tetrahedral finite elements (residual-kernel hyperreduction)")
        .add< HyperReducedTetrahedronFEMForceFieldResidualKernel<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronFEMForceFieldResidualKernel<Vec3Types>;
} // namespace sofa::component::forcefield
