/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTRIANGLEFEMFORCEFIELDKPCA_CPP
#include <ModelOrderReduction/component/forcefield/HyperReducedTriangleFEMForceFieldKPCA.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <ModelOrderReduction/config.h>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(HyperReducedTriangleFEMForceFieldKPCA)

void registerHyperReducedTriangleFEMForceFieldKPCA(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Triangular finite elements (kPCA-based hyperreduction)")
        .add< HyperReducedTriangleFEMForceFieldKPCA<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedTriangleFEMForceFieldKPCA<Vec3Types>;
} // namespace sofa::component::forcefield
