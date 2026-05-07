/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDRESTSHAPESPRINGSFORCEFIELDAE_CPP

#include <ModelOrderReduction/component/forcefield/HyperReducedRestShapeSpringsForceFieldAE.inl>

#include <sofa/helper/visual/DrawTool.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::solidmechanics::spring
{

using namespace sofa::defaulttype;


void registerHyperReducedRestShapeSpringsForceFieldAE(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Rest-shape springs (autoencoder-based hyperreduction)")
        .add< HyperReducedRestShapeSpringsForceFieldAE<Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API HyperReducedRestShapeSpringsForceFieldAE<Vec3Types>;
} // namespace sofa::component::solidmechanics::spring
