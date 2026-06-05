/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/component/mapping/ResidualKernelMapping.h>
#include <sofa/component/mapping/linear/LinearMapping.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
ResidualKernelMapping<TIn, TOut>::ResidualKernelMapping()
    : Parent()
    , d_residualKernelBundle(initData(&d_residualKernelBundle, "residualKernelBundle",
        "Path to residual-kernel bundle directory. REQUIRED."))
{
    d_residualKernelBundle.setPathType(sofa::core::objectmodel::PathType::DIRECTORY);
}

template <class TIn, class TOut>
void ResidualKernelMapping<TIn, TOut>::init()
{
    const unsigned n_in = this->fromModel->getSize();
    const unsigned n_out = this->toModel->getSize();

    m_projector = std::make_unique<sofa::component::kernel::ResidualKernelProjector>();
    m_projector->loadFromBundle(d_residualKernelBundle.getValue());

    msg_info(this) << "loaded residual-kernel bundle "
                   << d_residualKernelBundle.getValue()
                   << "  3N=" << m_projector->nbDofs()
                   << "  m=" << m_projector->nbModes()
                   << "  r=" << m_projector->nbResidualModes()
                   << "  T=" << m_projector->nbTrain()
                   << "  (mstate dofs: in=" << n_in << "  out=" << n_out << ")";

    if (m_projector->nbModes() != n_in)
    {
        msg_error(this) << "Bundle has " << m_projector->nbModes()
                        << " modes but mstate requests " << n_in;
    }

    m_q_prev.setZero(n_in);
    sofa::component::mapping::linear::LinearMapping<TIn, TOut>::init();
    m_J_dirty = true;
}

template <class TIn, class TOut>
void ResidualKernelMapping<TIn, TOut>::ensureJ()
{
    if (!m_J_dirty) return;
    m_J_cached = m_projector->J(m_q_prev);
    m_J_dirty = false;
}

template <class TIn, class TOut>
void ResidualKernelMapping<TIn, TOut>::reset()
{
    m_J_dirty = true;
    Parent::reset();
}

template <class TIn, class TOut>
void ResidualKernelMapping<TIn, TOut>::apply(const core::MechanicalParams* /*mparams*/,
                                             Data<VecCoord>& dOut,
                                             const Data<InVecCoord>& dIn)
{
    SCOPED_TIMER("apply in ResidualKernelMapping");

    helper::ReadAccessor<Data<InVecCoord>> in = dIn;
    const Eigen::Index m = static_cast<Eigen::Index>(in.size());
    Eigen::VectorXd q(m);
    for (Eigen::Index j = 0; j < m; ++j)
        q(j) = in[j][0];

    const Eigen::VectorXd u = m_projector->decode(q);
    const Eigen::Index N = static_cast<Eigen::Index>(u.size() / 3);

    helper::WriteOnlyAccessor<Data<VecCoord>> out = dOut;
    out.resize(N);
    for (Eigen::Index i = 0; i < N; ++i)
        out[i] = Coord(u(3 * i + 0), u(3 * i + 1), u(3 * i + 2));

    m_q_prev = q;
    m_J_dirty = true;
}

template <class TIn, class TOut>
void ResidualKernelMapping<TIn, TOut>::applyJ(const core::MechanicalParams* /*mparams*/,
                                              Data<VecDeriv>& dOut,
                                              const Data<InVecDeriv>& dIn)
{
    SCOPED_TIMER("applyJ in ResidualKernelMapping");

    helper::WriteOnlyAccessor<Data<VecDeriv>> out = dOut;
    helper::ReadAccessor<Data<InVecDeriv>> in = dIn;

    ensureJ();
    const Eigen::Index N = static_cast<Eigen::Index>(m_J_cached.rows() / 3);
    const Eigen::Index m = static_cast<Eigen::Index>(in.size());

    Eigen::VectorXd dq(m);
    for (Eigen::Index j = 0; j < m; ++j)
        dq(j) = in[j][0];

    const Eigen::VectorXd du = m_J_cached * dq;
    out.resize(N);
    for (Eigen::Index i = 0; i < N; ++i)
        out[i] = Deriv(du(3 * i + 0), du(3 * i + 1), du(3 * i + 2));
}

template <class TIn, class TOut>
void ResidualKernelMapping<TIn, TOut>::applyJT(const core::MechanicalParams* /*mparams*/,
                                               Data<InVecDeriv>& dOut,
                                               const Data<VecDeriv>& dIn)
{
    SCOPED_TIMER("applyJT in ResidualKernelMapping");

    helper::WriteAccessor<Data<InVecDeriv>> out = dOut;
    helper::ReadAccessor<Data<VecDeriv>> in = dIn;

    ensureJ();
    const Eigen::Index N = static_cast<Eigen::Index>(in.size());
    Eigen::VectorXd f(3 * N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            f(3 * i + c) = in[i][c];

    const Eigen::VectorXd df = m_J_cached.transpose() * f;
    for (Eigen::Index j = 0; j < df.size(); ++j)
        out[j][0] += df(j);
}

template <class TIn, class TOut>
void ResidualKernelMapping<TIn, TOut>::applyJT(const core::ConstraintParams* /*cparams*/,
                                               Data<InMatrixDeriv>& dOut,
                                               const Data<MatrixDeriv>& dIn)
{
    SCOPED_TIMER("applyJT(constraint) in ResidualKernelMapping");

    InMatrixDeriv& out = *dOut.beginEdit();
    const MatrixDeriv& in = dIn.getValue();

    ensureJ();
    const Eigen::Index m = m_J_cached.cols();

    for (auto rowIt = in.begin(), rowEnd = in.end(); rowIt != rowEnd; ++rowIt)
    {
        auto colIt = rowIt.begin();
        auto colItEnd = rowIt.end();
        typename TIn::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

        while (colIt != colItEnd)
        {
            const Eigen::Index idx = colIt.index();
            const auto val = colIt.val();
            for (Eigen::Index j = 0; j < m; ++j)
            {
                InDeriv data;
                data[0] = m_J_cached(3 * idx + 0, j) * val[0]
                       + m_J_cached(3 * idx + 1, j) * val[1]
                       + m_J_cached(3 * idx + 2, j) * val[2];
                o.addCol(j, data);
            }
            ++colIt;
        }
    }

    dOut.endEdit();
}

} // namespace sofa::component::mapping
