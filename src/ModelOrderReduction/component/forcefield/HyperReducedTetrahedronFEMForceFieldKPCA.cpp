/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONFEMFORCEFIELDKPCA_CPP
#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronFEMForceFieldKPCA.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <ModelOrderReduction/config.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(HyperReducedTetrahedronFEMForceFieldKPCA)

void registerHyperReducedTetrahedronFEMForceFieldKPCA(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Tetrahedral finite elements (kPCA-based hyperreduction)")
        .add< HyperReducedTetrahedronFEMForceFieldKPCA<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronFEMForceFieldKPCA<Vec3Types>;
} // namespace sofa::component::forcefield
