/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDRESTSHAPESPRINGSFORCEFIELDKPCA_CPP

#include <ModelOrderReduction/component/forcefield/HyperReducedRestShapeSpringsForceFieldKPCA.inl>

#include <sofa/helper/visual/DrawTool.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::solidmechanics::spring
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(HyperReducedRestShapeSpringsForceFieldKPCA)

void registerHyperReducedRestShapeSpringsForceFieldKPCA(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Rest-shape springs (kPCA-based hyperreduction)")
        .add< HyperReducedRestShapeSpringsForceFieldKPCA<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedRestShapeSpringsForceFieldKPCA<Vec3Types>;
} // namespace sofa::component::solidmechanics::spring
