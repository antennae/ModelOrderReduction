/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDKPCA_CPP

#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA)

void registerHyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Generic Tetrahedral hyperelastic finite elements (kPCA-based hyperreduction)")
        .add< HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA<Vec3Types>;
} // namespace sofa::component::forcefield
