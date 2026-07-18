/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDQUADRATICMANIFOLD_CPP

#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold)

void registerHyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Generic Tetrahedral hyperelastic finite elements (quadratic-manifold-based hyperreduction)")
        .add< HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold<Vec3Types>;
} // namespace sofa::component::forcefield
