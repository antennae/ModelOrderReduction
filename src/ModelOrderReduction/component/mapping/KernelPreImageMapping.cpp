/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#define SOFA_COMPONENT_MAPPING_KERNELPREIMAGEMAPPING_CPP
#include <ModelOrderReduction/component/mapping/KernelPreImageMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mapping
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(KernelPreImageMapping)

void registerKernelPreImageMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Explicit kPCA pre-image reduced-model mapping (exact decode u=Ψ(q) + "
        "implicit-function-theorem Jacobian K_pre⁻¹(J̃+ηM·J_init) + geometric "
        "stiffness). Sibling of KernelPCAMapping.")
        .add< KernelPreImageMapping<Vec1Types, Vec3Types> >());
}

template class SOFA_MODELORDERREDUCTION_API KernelPreImageMapping<Vec1Types, Vec3Types>;

} // namespace sofa::component::mapping
