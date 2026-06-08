/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#include <ModelOrderReduction/component/kernel/DecoderProjector.h>

namespace sofa::component::kernel
{

DecoderProjector::VectorXd DecoderProjector::project_element(
    const Eigen::Ref<const VectorXd>& q,
    const std::vector<unsigned int>& indexList,
    const Eigen::Ref<const VectorXd>& contrib) const
{
    const unsigned V = static_cast<unsigned>(indexList.size());
    if (static_cast<unsigned>(contrib.size()) != 3 * V)
        throw std::runtime_error(
            "DecoderProjector::project_element: contrib size != 3 * indexList size");

    VectorXd full = VectorXd::Zero(nbDofs());
    for (unsigned i = 0; i < V; ++i)
    {
        if (3u * indexList[i] + 2u >= nbDofs())
            throw std::runtime_error(
                "DecoderProjector::project_element: node index " +
                std::to_string(indexList[i]) + " out of range (nbDofs=" +
                std::to_string(nbDofs()) + ")");
        const unsigned dof0 = 3 * indexList[i];
        full(dof0 + 0) = contrib(3 * i + 0);
        full(dof0 + 1) = contrib(3 * i + 1);
        full(dof0 + 2) = contrib(3 * i + 2);
    }
    return project_force(q, full);
}

DecoderProjector::VectorXd DecoderProjector::encode(
    const Eigen::Ref<const VectorXd>& /*u_disp*/) const
{
    throw std::runtime_error("encode not supported by this DecoderProjector");
}

const DecoderProjector::MatrixXd& DecoderProjector::rigidModes() const
{
    static const MatrixXd empty;
    return empty;
}

} // namespace sofa::component::kernel
