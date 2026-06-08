/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONFEMFORCEFIELDQUADRATICMANIFOLD_CPP
#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronFEMForceFieldQuadraticManifold.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <ModelOrderReduction/config.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;


void registerHyperReducedTetrahedronFEMForceFieldQuadraticManifold(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Tetrahedral finite elements (quadratic-manifold hyperreduction)")
        .add< HyperReducedTetrahedronFEMForceFieldQuadraticManifold<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronFEMForceFieldQuadraticManifold<Vec3Types>;
} // namespace sofa::component::forcefield
