/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperQuadraticManifold.h>

namespace modelorderreduction
{

HyperReducedHelperQuadraticManifold::HyperReducedHelperQuadraticManifold()
    : d_quadraticManifoldBundle(initData(&d_quadraticManifoldBundle,
          std::string("residual_kernel"), "quadraticManifoldBundle",
          "Path to quadratic-manifold bundle directory"))
{
    d_quadraticManifoldBundle.setGroup("HyperReduction");
}

} // namespace modelorderreduction
