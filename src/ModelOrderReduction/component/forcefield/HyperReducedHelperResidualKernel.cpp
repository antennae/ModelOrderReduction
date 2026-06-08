/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperResidualKernel.h>

namespace modelorderreduction
{

HyperReducedHelperResidualKernel::HyperReducedHelperResidualKernel()
    : d_residualKernelBundle(initData(&d_residualKernelBundle,
          std::string("residual_kernel"), "residualKernelBundle",
          "Path to residual-kernel bundle directory"))
{
    d_residualKernelBundle.setGroup("HyperReduction");
}

} // namespace modelorderreduction
