/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   Quadratic-manifold hyperreduction helper — thin shim over                 *
*   HyperReducedHelperDecoder (supplies the projector type + bundle path).    *
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/forcefield/HyperReducedHelperDecoder.h>
#include <ModelOrderReduction/component/kernel/QuadraticManifoldProjector.h>

#include <sofa/core/objectmodel/Data.h>

#include <memory>
#include <string>

namespace modelorderreduction
{

using sofa::core::objectmodel::Data;

class SOFA_MODELORDERREDUCTION_API HyperReducedHelperQuadraticManifold
    : public HyperReducedHelperDecoder
{
public:
    SOFA_CLASS(HyperReducedHelperQuadraticManifold, HyperReducedHelperDecoder);

    Data<std::string> d_quadraticManifoldBundle;

    HyperReducedHelperQuadraticManifold();

protected:
    std::unique_ptr<sofa::component::kernel::DecoderProjector>
    createProjector() const override
    {
        return std::make_unique<sofa::component::kernel::QuadraticManifoldProjector>();
    }
    std::string bundlePath() const override
    {
        return d_quadraticManifoldBundle.getValue();
    }
};

} // namespace modelorderreduction
