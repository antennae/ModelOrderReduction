/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/component/mapping/AEMapping.h>
#include <sofa/component/mapping/linear/LinearMapping.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
AEMapping<TIn, TOut>::AEMapping()
    : Parent()
    , d_aeBundle(initData(&d_aeBundle, "aeBundle",
        "Path to the AE bundle directory (X0.txt, col_std.txt, decoder.ts.pt; "
        "optional encoder.ts.pt, rigid_modes.txt). REQUIRED."))
{
    d_aeBundle.setPathType(sofa::core::objectmodel::PathType::DIRECTORY);
}

template <class TIn, class TOut>
void AEMapping<TIn, TOut>::init()
{
    const unsigned n_in  = this->fromModel->getSize();
    const unsigned n_out = this->toModel->getSize();

    m_projector = std::make_unique<sofa::component::kernel::AEProjector>();
    m_projector->loadFromBundle(d_aeBundle.getValue());

    msg_info(this) << "loaded AE bundle " << d_aeBundle.getValue()
                   << "  3N="     << m_projector->nbDofs()
                   << "  m="      << m_projector->nbModes()
                   << "  nbRigid=" << m_projector->nbRigid()
                   << "  (mstate dofs: in=" << n_in << "  out=" << n_out << ")";

    const unsigned nbTotal = m_projector->nbModes() + m_projector->nbRigid();
    if (nbTotal < n_in)
    {
        msg_error(this) << "Bundle has " << nbTotal
                        << " modes (m_def=" << m_projector->nbModes()
                        << " + nbRigid=" << m_projector->nbRigid()
                        << ") but mstate requests " << n_in;
    }

    // Reset incremental state.
    m_q_prev.setZero(n_in);

    // Cache rigid translation columns (empty if no rigid modes).
    m_PhiT = m_projector->rigidModes();

    // Skip Parent::init()'s modesPath logic.
    sofa::component::mapping::linear::LinearMapping<TIn, TOut>::init();

    // J(q) is always state-dependent for AE; rebuild on first applyJ/applyJT.
    m_J_dirty = true;

    const auto& rot = this->d_rotation.getValue();
    if (rot[0] != 0.0 || rot[1] != 0.0 || rot[2] != 0.0)
        msg_warning(this) << "rotation Data is not yet wired for AE mapping; ignored.";
}

template <class TIn, class TOut>
void AEMapping<TIn, TOut>::ensureJ()
{
    if (!m_J_dirty) return;

    const Eigen::Index nbRigid = static_cast<Eigen::Index>(m_projector->nbRigid());
    const Eigen::Index nbDef   = static_cast<Eigen::Index>(m_projector->nbModes());
    const Eigen::Index N3      = static_cast<Eigen::Index>(m_projector->nbDofs());

    // Decoder Jacobian is a function of latent q, not of input-space u.
    // Use m_q_prev as the linearization point — it holds the q paired with
    // the current toModel position.
    const Eigen::VectorXd q_def_prev = (nbRigid > 0) ? m_q_prev.tail(nbDef) : m_q_prev;
    const Eigen::MatrixXd J_def = m_projector->J(q_def_prev);

    if (nbRigid > 0)
    {
        m_J_cached.resize(N3, nbRigid + nbDef);
        m_J_cached.leftCols(nbRigid) = m_PhiT;
        m_J_cached.rightCols(nbDef)  = J_def;
    }
    else
    {
        m_J_cached = J_def;
    }
    m_J_dirty = false;
}

template <class TIn, class TOut>
void AEMapping<TIn, TOut>::reset()
{
    // Same rationale as KernelPCAMapping::reset — leave m_q_prev so Δq = 0
    // across the reset and the incremental decode stays consistent.
    m_J_dirty = true;
    Parent::reset();
}


template <class TIn, class TOut>
void AEMapping<TIn, TOut>::apply(const core::MechanicalParams* /*mparams*/,
                                 Data<VecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    SCOPED_TIMER("apply in AEMapping");

    helper::ReadAccessor<Data<InVecCoord>> in = dIn;

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

    // ensureJ uses m_q_prev — the q paired with u_old.
    ensureJ();
    const Eigen::VectorXd du = m_J_cached * dq;

    helper::WriteOnlyAccessor<Data<VecCoord>> out = dOut;
    out.resize(N);
    for (Eigen::Index i = 0; i < N; ++i)
        out[i] = Coord(u_old(3 * i + 0) + du(3 * i + 0),
                       u_old(3 * i + 1) + du(3 * i + 1),
                       u_old(3 * i + 2) + du(3 * i + 2));

    m_q_prev = q_new;
    m_J_dirty = true;       // q just changed; rebuild J at the new linearisation point next call.
}


template <class TIn, class TOut>
void AEMapping<TIn, TOut>::applyJ(const core::MechanicalParams* /*mparams*/,
                                  Data<VecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    SCOPED_TIMER("applyJ in AEMapping");

    helper::WriteOnlyAccessor<Data<VecDeriv>> out = dOut;
    helper::ReadAccessor<Data<InVecDeriv>>    in  = dIn;

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
void AEMapping<TIn, TOut>::applyJT(const core::MechanicalParams* /*mparams*/,
                                   Data<InVecDeriv>& dOut, const Data<VecDeriv>& dIn)
{
    SCOPED_TIMER("applyJT in AEMapping");

    helper::WriteAccessor<Data<InVecDeriv>> out = dOut;
    helper::ReadAccessor<Data<VecDeriv>>    in  = dIn;

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
void AEMapping<TIn, TOut>::applyJT(const core::ConstraintParams* /*cparams*/,
                                   Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn)
{
    SCOPED_TIMER("applyJT(constraint) in AEMapping");

    InMatrixDeriv& out = *dOut.beginEdit();
    const MatrixDeriv& in = dIn.getValue();

    ensureJ();
    const Eigen::MatrixXd& J_q = m_J_cached;
    const Eigen::Index m = J_q.cols();

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

        while (colIt != colItEnd)
        {
            const Eigen::Index idx = colIt.index();
            const auto val = colIt.val();
            for (Eigen::Index j = 0; j < m; ++j)
            {
                InDeriv data;
                data[0] = J_q(3 * idx + 0, j) * val[0]
                       + J_q(3 * idx + 1, j) * val[1]
                       + J_q(3 * idx + 2, j) * val[2];
                o.addCol(j, data);
            }
            ++colIt;
        }
    }

    dOut.endEdit();
}

} // namespace sofa::component::mapping
