/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONFEMFORCEFIELDAE_CPP
#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronFEMForceFieldAE.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <ModelOrderReduction/config.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;


void registerHyperReducedTetrahedronFEMForceFieldAE(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Tetrahedral finite elements (autoencoder-based hyperreduction)")
        .add< HyperReducedTetrahedronFEMForceFieldAE<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronFEMForceFieldAE<Vec3Types>;
} // namespace sofa::component::forcefield
