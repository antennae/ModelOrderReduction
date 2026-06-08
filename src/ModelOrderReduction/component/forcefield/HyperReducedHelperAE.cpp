/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperAE.h>

namespace modelorderreduction
{

HyperReducedHelperAE::HyperReducedHelperAE()
    : d_aeBundle(initData(&d_aeBundle, std::string("ae"), "aeBundle",
          "Path to the AE bundle directory (decoder.ts.pt, X0.txt, col_std.txt; optional encoder.ts.pt, rigid_modes.txt)"))
{
    d_aeBundle.setGroup("HyperReduction");
}

} // namespace modelorderreduction
