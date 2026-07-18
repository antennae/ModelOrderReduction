/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDRESIDUALKERNEL_CPP

#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronHyperelasticityFEMForceFieldResidualKernel.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(HyperReducedTetrahedronHyperelasticityFEMForceFieldResidualKernel)

void registerHyperReducedTetrahedronHyperelasticityFEMForceFieldResidualKernel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Generic Tetrahedral hyperelastic finite elements (residual-kernel-based hyperreduction)")
        .add< HyperReducedTetrahedronHyperelasticityFEMForceFieldResidualKernel<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronHyperelasticityFEMForceFieldResidualKernel<Vec3Types>;
} // namespace sofa::component::forcefield
