/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/component/mapping/KernelPCAMapping.h>
#include <sofa/component/mapping/linear/LinearMapping.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
KernelPCAMapping<TIn, TOut>::KernelPCAMapping()
    : Parent()
    , d_kernelBundle(initData(&d_kernelBundle, "kernelBundle",
        "Path to the kPCA bundle directory (X0.txt, snapshots.txt, alpha.txt, kernel.txt). REQUIRED."))
{
    d_kernelBundle.setPathType(sofa::core::objectmodel::PathType::DIRECTORY);
}

template <class TIn, class TOut>
void KernelPCAMapping<TIn, TOut>::init()
{
    const unsigned n_in  = this->fromModel->getSize();
    const unsigned n_out = this->toModel->getSize();

    m_projector = sofa::component::kernel::loadKernelProjectorFromBundle(
        d_kernelBundle.getValue());
    msg_info(this) << "loaded kPCA bundle " << d_kernelBundle.getValue()
                   << "  kernel=" << m_projector->kernelName()
                   << "  3N="     << m_projector->nbDofs()
                   << "  T="      << m_projector->nbSnapshots()
                   << "  m="      << m_projector->nbModes()
                   << "  (mstate dofs: in="<< n_in << "  out=" << n_out << ")";

    if (m_projector->nbModes() < n_in)
    {
        msg_error(this) << "Bundle has " << m_projector->nbModes()
                        << " modes but mstate requests " << n_in;
    }

    // Reset incremental state: u starts at toModel's rest (set by SOFA);
    // q_prev starts at zero so the first apply produces u_new = u_rest + J(u_rest)·q.
    m_q_prev.setZero(n_in);

    // Skip Parent::init() — we don't want it to try loading modesPath.
    sofa::component::mapping::linear::LinearMapping<TIn, TOut>::init();

    const auto& rot = this->d_rotation.getValue();
    if (rot[0] != 0.0 || rot[1] != 0.0 || rot[2] != 0.0)
        msg_warning(this) << "rotation Data is not yet wired for kPCA mapping; ignored.";
}

template <class TIn, class TOut>
void KernelPCAMapping<TIn, TOut>::reset()
{
    // Intentionally do NOT zero m_q_prev: SOFA's reset visitor doesn't
    // resync our toModel's position to its rest, so leaving m_q_prev as
    // "the q that produced the current u" keeps Δq = 0 across the reset
    // and the incremental decode stays consistent. q_prev/u_old will be
    // updated together once a real animation step changes q.
    Parent::reset();
}


template <class TIn, class TOut>
void KernelPCAMapping<TIn, TOut>::apply(const core::MechanicalParams* /*mparams*/,
                                        Data<VecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    SCOPED_TIMER("apply in KernelPCAMapping");

    helper::ReadAccessor<Data<InVecCoord>> in = dIn;

    // Snapshot u_old (pre-write) by copy — the WriteOnlyAccessor below
    // is going to overwrite the same buffer.
    const VecCoord u_old_vec = this->toModel->read(core::vec_id::read_access::position)->getValue();
    const Eigen::Index N = static_cast<Eigen::Index>(u_old_vec.size());
    const Eigen::Index m = static_cast<Eigen::Index>(in.size());

    Eigen::VectorXd u_old(3 * N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            u_old(3 * i + c) = u_old_vec[i][c];

    Eigen::VectorXd q_new(m);
    for (Eigen::Index j = 0; j < m; ++j)
        q_new(j) = in[j][0];

    if (m_q_prev.size() != m) m_q_prev.setZero(m);
    const Eigen::VectorXd dq = q_new - m_q_prev;

    const Eigen::VectorXd du = m_projector->applyJ(u_old, dq);   // J(u_old) · Δq

    helper::WriteOnlyAccessor<Data<VecCoord>> out = dOut;
    out.resize(N);
    for (Eigen::Index i = 0; i < N; ++i)
        out[i] = Coord(u_old(3 * i + 0) + du(3 * i + 0),
                       u_old(3 * i + 1) + du(3 * i + 1),
                       u_old(3 * i + 2) + du(3 * i + 2));

    m_q_prev = q_new;
}


template <class TIn, class TOut>
void KernelPCAMapping<TIn, TOut>::applyJ(const core::MechanicalParams* /*mparams*/,
                                         Data<VecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    SCOPED_TIMER("applyJ in KernelPCAMapping");

    helper::WriteOnlyAccessor<Data<VecDeriv>> out = dOut;
    helper::ReadAccessor<Data<InVecDeriv>>    in  = dIn;

    const VecCoord u_vec = this->toModel->read(core::vec_id::read_access::position)->getValue();
    const Eigen::Index N = static_cast<Eigen::Index>(u_vec.size());
    const Eigen::Index m = static_cast<Eigen::Index>(in.size());

    Eigen::VectorXd u(3 * N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            u(3 * i + c) = u_vec[i][c];

    Eigen::VectorXd dq(m);
    for (Eigen::Index j = 0; j < m; ++j)
        dq(j) = in[j][0];

    const Eigen::VectorXd du = m_projector->applyJ(u, dq);

    out.resize(N);
    for (Eigen::Index i = 0; i < N; ++i)
        out[i] = Deriv(du(3 * i + 0), du(3 * i + 1), du(3 * i + 2));
}


template <class TIn, class TOut>
void KernelPCAMapping<TIn, TOut>::applyJT(const core::MechanicalParams* /*mparams*/,
                                          Data<InVecDeriv>& dOut, const Data<VecDeriv>& dIn)
{
    SCOPED_TIMER("applyJT in KernelPCAMapping");

    helper::WriteAccessor<Data<InVecDeriv>> out = dOut;
    helper::ReadAccessor<Data<VecDeriv>>    in  = dIn;

    const VecCoord u_vec = this->toModel->read(core::vec_id::read_access::position)->getValue();
    const Eigen::Index N = static_cast<Eigen::Index>(in.size());

    Eigen::VectorXd u(3 * N);
    Eigen::VectorXd f(3 * N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
        {
            u(3 * i + c) = u_vec[i][c];
            f(3 * i + c) = in[i][c];
        }

    const Eigen::VectorXd df = m_projector->project_force(u, f);  // (m,)

    for (Eigen::Index j = 0; j < df.size(); ++j)
        out[j][0] += df(j);
}


template <class TIn, class TOut>
void KernelPCAMapping<TIn, TOut>::applyJT(const core::ConstraintParams* /*cparams*/,
                                          Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn)
{
    InMatrixDeriv& out = *dOut.beginEdit();
    const MatrixDeriv& in = dIn.getValue();

    // Build J(u) once and use for every constraint row.
    const VecCoord u_vec = this->toModel->read(core::vec_id::read_access::position)->getValue();
    const Eigen::Index N = static_cast<Eigen::Index>(u_vec.size());
    Eigen::VectorXd u(3 * N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            u(3 * i + c) = u_vec[i][c];

    const Eigen::MatrixXd J_u = m_projector->J(u);   // (3N, m)
    const Eigen::Index m = J_u.cols();

    for (auto rowIt = in.begin(), rowEnd = in.end(); rowIt != rowEnd; ++rowIt)
    {
        auto colIt    = rowIt.begin();
        auto colItEnd = rowIt.end();
        if (colIt == colItEnd)
        {
            msg_fatal(this) << "constraint row is empty — not implemented";
            continue;
        }

        typename TIn::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

        // Each non-zero entry of the row is a 3-vector at vertex `colIt.index()`;
        // it contributes  J_u.row(3·idx:3·idx+3)ᵀ · val  to every reduced mode.
        while (colIt != colItEnd)
        {
            const Eigen::Index idx = colIt.index();
            const auto val = colIt.val();   // Deriv (3-vector)
            for (Eigen::Index j = 0; j < m; ++j)
            {
                InDeriv data;
                data[0] = J_u(3 * idx + 0, j) * val[0]
                       + J_u(3 * idx + 1, j) * val[1]
                       + J_u(3 * idx + 2, j) * val[2];
                o.addCol(j, data);
            }
            ++colIt;
        }
    }

    dOut.endEdit();
}

} // namespace sofa::component::mapping
