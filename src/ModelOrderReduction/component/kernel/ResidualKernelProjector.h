/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                         residual-kernel decoder primitives                  *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/config.h>

#include <Eigen/Core>
#include <string>

namespace sofa::component::kernel
{

class SOFA_MODELORDERREDUCTION_API ResidualKernelProjector
{
public:
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    void loadFromBundle(const std::string& bundle_dir);

    VectorXd decode(const Eigen::Ref<const VectorXd>& q) const;
    MatrixXd J(const Eigen::Ref<const VectorXd>& q) const;
    VectorXd applyJ(const Eigen::Ref<const VectorXd>& q,
                    const Eigen::Ref<const VectorXd>& dq) const;
    VectorXd project_force(const Eigen::Ref<const VectorXd>& q,
                           const Eigen::Ref<const VectorXd>& f) const;

    const VectorXd& mean() const { return m_mean; }
    const VectorXd& rest() const { return m_rest; }
    const MatrixXd& primaryBasis() const { return m_primaryBasis; }
    const MatrixXd& residualBasis() const { return m_residualBasis; }
    const MatrixXd& trainQ() const { return m_trainQ; }
    const MatrixXd& krrWeights() const { return m_krrWeights; }
    double lengthScale() const { return m_lengthScale; }
    double ridge() const { return m_ridge; }

    unsigned nbDofs() const { return static_cast<unsigned>(m_mean.size()); }
    unsigned nbModes() const { return static_cast<unsigned>(m_primaryBasis.cols()); }
    unsigned nbResidualModes() const { return static_cast<unsigned>(m_residualBasis.cols()); }
    unsigned nbTrain() const { return static_cast<unsigned>(m_trainQ.rows()); }

    static const std::string& projectorName();

private:
    VectorXd kernelVector(const Eigen::Ref<const VectorXd>& q) const;

    VectorXd m_mean;
    VectorXd m_rest;
    MatrixXd m_primaryBasis;
    MatrixXd m_residualBasis;
    MatrixXd m_trainQ;
    MatrixXd m_krrWeights;
    double m_lengthScale = 1.0;
    double m_ridge = 0.0;
};

} // namespace sofa::component::kernel
