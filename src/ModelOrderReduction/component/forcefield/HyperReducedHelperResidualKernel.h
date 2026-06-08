/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   Residual-kernel hyperreduction helper — thin shim over                    *
*   HyperReducedHelperDecoder (supplies the projector type + bundle path).    *
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/forcefield/HyperReducedHelperDecoder.h>
#include <ModelOrderReduction/component/kernel/ResidualKernelProjector.h>

#include <sofa/core/objectmodel/Data.h>

#include <memory>
#include <string>

namespace modelorderreduction
{

using sofa::core::objectmodel::Data;

class SOFA_MODELORDERREDUCTION_API HyperReducedHelperResidualKernel
    : public HyperReducedHelperDecoder
{
public:
    SOFA_CLASS(HyperReducedHelperResidualKernel, HyperReducedHelperDecoder);

    Data<std::string> d_residualKernelBundle;

    HyperReducedHelperResidualKernel();

protected:
    std::unique_ptr<sofa::component::kernel::DecoderProjector>
    createProjector() const override
    {
        return std::make_unique<sofa::component::kernel::ResidualKernelProjector>();
    }
    std::string bundlePath() const override
    {
        return d_residualKernelBundle.getValue();
    }
};

} // namespace modelorderreduction
