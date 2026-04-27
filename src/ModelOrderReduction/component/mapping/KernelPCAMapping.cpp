/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_MAPPING_KERNELPCAMAPPING_CPP
#include <ModelOrderReduction/component/mapping/KernelPCAMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mapping
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(KernelPCAMapping)

void registerKernelPCAMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Reduced model (kPCA-based; loads a bundle instead of modes.txt)")
        .add< KernelPCAMapping<Vec1Types, Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API KernelPCAMapping<Vec1Types, Vec3Types>;

} // namespace sofa::component::mapping
