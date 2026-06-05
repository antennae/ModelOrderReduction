/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_MAPPING_RESIDUALKERNELMAPPING_CPP
#include <ModelOrderReduction/component/mapping/ResidualKernelMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mapping
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ResidualKernelMapping)

void registerResidualKernelMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Reduced model with explicit residual-kernel decoder")
        .add< ResidualKernelMapping<Vec1Types, Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API ResidualKernelMapping<Vec1Types, Vec3Types>;

} // namespace sofa::component::mapping
