#pragma once

#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/kernel/DecoderProjector.h>

#include <Eigen/Core>
#include <string>

namespace sofa::component::kernel
{

class SOFA_MODELORDERREDUCTION_API QuadraticManifoldProjector : public DecoderProjector
{
public:
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    void loadFromBundle(const std::string& bundleDir) override;

    VectorXd decode(const Eigen::Ref<const VectorXd>& q) const override;
    MatrixXd J(const Eigen::Ref<const VectorXd>& q) const override;

    const VectorXd& mean() const { return m_mean; }
    const VectorXd& rest() const { return m_rest; }
    const MatrixXd& linearBasis() const { return m_linearBasis; }
    const MatrixXd& quadraticBasis() const { return m_quadraticBasis; }
    double ridge() const { return m_ridge; }

    unsigned nbDofs() const override { return static_cast<unsigned>(m_mean.size()); }
    unsigned nbModes() const override
    {
        return static_cast<unsigned>(m_linearBasis.cols());
    }

    static const std::string& projectorName();

private:
    void checkQ(const Eigen::Ref<const VectorXd>& q) const;

    VectorXd m_mean;
    VectorXd m_rest;
    MatrixXd m_linearBasis;
    MatrixXd m_quadraticBasis;
    double m_ridge = 0.0;
};

} // namespace sofa::component::kernel
