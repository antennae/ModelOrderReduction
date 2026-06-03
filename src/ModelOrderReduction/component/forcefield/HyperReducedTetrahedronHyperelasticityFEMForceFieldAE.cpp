/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDAE_CPP

#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronHyperelasticityFEMForceFieldAE.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(HyperReducedTetrahedronHyperelasticityFEMForceFieldAE)

void registerHyperReducedTetrahedronHyperelasticityFEMForceFieldAE(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Generic Tetrahedral hyperelastic finite elements (autoencoder-based hyperreduction)")
        .add< HyperReducedTetrahedronHyperelasticityFEMForceFieldAE<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronHyperelasticityFEMForceFieldAE<Vec3Types>;
} // namespace sofa::component::forcefield
