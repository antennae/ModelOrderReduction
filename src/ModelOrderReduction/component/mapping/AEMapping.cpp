/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_MAPPING_AEMAPPING_CPP
#include <ModelOrderReduction/component/mapping/AEMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mapping
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(AEMapping)

void registerAEMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Reduced model (autoencoder-based; loads a TorchScript bundle).")
        .add< AEMapping<Vec1Types, Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API AEMapping<Vec1Types, Vec3Types>;

} // namespace sofa::component::mapping
