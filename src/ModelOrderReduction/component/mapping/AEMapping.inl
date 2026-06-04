/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/component/mapping/AEMapping.h>
#include <sofa/component/mapping/linear/LinearMapping.h>
#include <sofa/core/MechanicalParams.h>
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
    , d_geomEps(initData(&d_geomEps, 0.0, "geomEps",
        "FD step for the applyDJT geometric stiffness (∂J/∂q)ᵀf. "
        "<=0 disables geometric stiffness (exact decode + first-order tangent)."))
    , d_decodeMode(initData(&d_decodeMode, std::string("exact"), "decodeMode",
        "Position reconstruction: 'exact' (u = X0 + Ψ(q), drift-free, default) "
        "or 'incremental' (u_new = u_old + J(q_old)·Δq, kPCA-style, robust "
        "fallback when exact decode inverts elements at high latent dim)."))
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

    const std::string mode = d_decodeMode.getValue();
    if (mode != "exact" && mode != "incremental")
        msg_error(this) << "decodeMode must be 'exact' or 'incremental', got '"
                        << mode << "'; defaulting to exact.";
    m_incremental = (mode == "incremental");
    msg_info(this) << "decode mode: " << (m_incremental ? "incremental" : "exact");

    // Reset cached reduced coordinates.
    m_q_prev.setZero(n_in);

    // Cache rigid translation columns (empty if no rigid modes).
    m_PhiT = m_projector->rigidModes();

    // Skip Parent::init()'s modesPath logic.
    sofa::component::mapping::linear::LinearMapping<TIn, TOut>::init();

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
    m_J_dirty = true;
    Parent::reset();
}


template <class TIn, class TOut>
void AEMapping<TIn, TOut>::apply(const core::MechanicalParams* /*mparams*/,
                                 Data<VecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    SCOPED_TIMER("apply in AEMapping");

    helper::ReadAccessor<Data<InVecCoord>> in = dIn;

    const Eigen::Index m = static_cast<Eigen::Index>(in.size());

    Eigen::VectorXd q_new(m);
    for (Eigen::Index j = 0; j < m; ++j)
        q_new(j) = in[j][0];

    const Eigen::Index nbRigid = static_cast<Eigen::Index>(m_projector->nbRigid());
    const Eigen::Index nbDef   = static_cast<Eigen::Index>(m_projector->nbModes());
    const Eigen::Index N3      = static_cast<Eigen::Index>(m_projector->nbDofs());
    if (m != nbRigid + nbDef)
    {
        msg_error(this) << "input q size " << m << " does not match AE bundle size "
                        << (nbRigid + nbDef) << " (nbRigid=" << nbRigid
                        << ", nbDef=" << nbDef << ")";
        return;
    }

    const Eigen::Index N = N3 / 3;
    helper::WriteOnlyAccessor<Data<VecCoord>> out = dOut;
    out.resize(N);

    if (m_incremental)
    {
        // Incremental (kPCA-style): u_new = u_old + J(q_old)·Δq. Stays on the
        // linear tangent — drift-prone but never invokes the decoder's
        // off-manifold curvature, so robust where exact decode inverts.
        const VecCoord u_old = this->toModel->read(core::vec_id::read_access::position)->getValue();
        ensureJ();
        const Eigen::VectorXd du = m_J_cached * (q_new - m_q_prev);
        for (Eigen::Index i = 0; i < N; ++i)
            out[i] = Coord(u_old[i][0] + du(3 * i + 0),
                           u_old[i][1] + du(3 * i + 1),
                           u_old[i][2] + du(3 * i + 2));
    }
    else
    {
        // Exact decode: u = X0 + Φ_t q_t + Ψ(q_def). Drift-free.
        const Eigen::VectorXd q_def_new = (nbRigid > 0) ? q_new.tail(nbDef) : q_new;
        Eigen::VectorXd u = m_projector->decode(q_def_new);
        if (nbRigid > 0)
            u += m_PhiT * q_new.head(nbRigid);

        const Eigen::VectorXd& X0 = m_projector->X0();
        for (Eigen::Index i = 0; i < N; ++i)
            out[i] = Coord(X0(3 * i + 0) + u(3 * i + 0),
                           X0(3 * i + 1) + u(3 * i + 1),
                           X0(3 * i + 2) + u(3 * i + 2));
    }

    m_q_prev = q_new;
    m_J_dirty = true;
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
void AEMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams,
                                    core::MultiVecDerivId parentForceId,
                                    core::ConstMultiVecDerivId /*childForceId*/)
{
    const double eps = d_geomEps.getValue();
    if (eps <= 0.0) return;

    SCOPED_TIMER("applyDJT in AEMapping");
    ensureJ();

    helper::ReadAccessor<Data<VecDeriv>> childForce =
        *mparams->readF(this->toModel.get());
    helper::ReadAccessor<Data<InVecDeriv>> parentDx =
        *mparams->readDx(this->fromModel.get());
    helper::WriteAccessor<Data<InVecDeriv>> parentForce =
        *parentForceId[this->fromModel.get()].write();

    const Eigen::Index N = static_cast<Eigen::Index>(childForce.size());
    const Eigen::Index m = m_J_cached.cols();

    Eigen::VectorXd f(3 * N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            f(3 * i + c) = childForce[i][c];

    Eigen::VectorXd dq(m);
    for (Eigen::Index j = 0; j < m; ++j)
        dq(j) = parentDx[j][0];

    const Eigen::Index nbRigid = static_cast<Eigen::Index>(m_projector->nbRigid());
    const Eigen::Index nbDef   = static_cast<Eigen::Index>(m_projector->nbModes());
    const Eigen::Index N3      = static_cast<Eigen::Index>(m_projector->nbDofs());

    const Eigen::VectorXd qp = m_q_prev + eps * dq;
    const Eigen::VectorXd q_def_p = (nbRigid > 0) ? qp.tail(nbDef) : qp;
    const Eigen::MatrixXd J_def_p = m_projector->J(q_def_p);

    Eigen::MatrixXd Jp;
    if (nbRigid > 0)
    {
        Jp.resize(N3, nbRigid + nbDef);
        Jp.leftCols(nbRigid) = m_PhiT;
        Jp.rightCols(nbDef)  = J_def_p;
    }
    else
    {
        Jp = J_def_p;
    }

    const Eigen::VectorXd dgeo = ((Jp - m_J_cached) / eps).transpose() * f;
    const double kfactor = static_cast<double>(mparams->kFactor());
    for (Eigen::Index j = 0; j < m; ++j)
        parentForce[j][0] += kfactor * dgeo(j);
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
