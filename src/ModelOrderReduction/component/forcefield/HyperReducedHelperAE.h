/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   AE hyperreduction helper — thin shim over HyperReducedHelperDecoder       *
*   (supplies the AEProjector type + bundle path). Rigid-mode handling lives  *
*   in AEProjector (rigidModes()/nbRigid() overrides), consumed by the base.  *
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/forcefield/HyperReducedHelperDecoder.h>
#include <ModelOrderReduction/component/kernel/AEProjector.h>

#include <sofa/core/objectmodel/Data.h>

#include <memory>
#include <string>

namespace modelorderreduction
{

using sofa::core::objectmodel::Data;

class SOFA_MODELORDERREDUCTION_API HyperReducedHelperAE
    : public HyperReducedHelperDecoder
{
public:
    SOFA_CLASS(HyperReducedHelperAE, HyperReducedHelperDecoder);

    Data<std::string> d_aeBundle;

    HyperReducedHelperAE();

protected:
    std::unique_ptr<sofa::component::kernel::DecoderProjector>
    createProjector() const override
    {
        return std::make_unique<sofa::component::kernel::AEProjector>();
    }
    std::string bundlePath() const override
    {
        return d_aeBundle.getValue();
    }
};

} // namespace modelorderreduction
