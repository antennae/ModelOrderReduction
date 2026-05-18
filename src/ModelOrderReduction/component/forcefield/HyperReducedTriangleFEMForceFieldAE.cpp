/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTRIANGLEFEMFORCEFIELDAE_CPP
#include <ModelOrderReduction/component/forcefield/HyperReducedTriangleFEMForceFieldAE.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <ModelOrderReduction/config.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;


void registerHyperReducedTriangleFEMForceFieldAE(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Triangular finite elements (autoencoder-based hyperreduction)")
        .add< HyperReducedTriangleFEMForceFieldAE<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTriangleFEMForceFieldAE<Vec3Types>;
} // namespace sofa::component::forcefield
