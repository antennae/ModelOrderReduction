/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/component/mapping/KernelPreImageMapping.h>
#include <sofa/component/mapping/linear/LinearMapping.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
KernelPreImageMapping<TIn, TOut>::KernelPreImageMapping()
    : Parent()
    , d_kernelBundle(initData(&d_kernelBundle, "kernelBundle",
        "Path to the kPCA bundle directory (X0, snapshots, alpha, kernel + "
        "preimage_config.json, mass_diagonal.txt). REQUIRED."))
    , d_eta(initData(&d_eta, -1.0, "eta",
        "Override bundle mechanical-reg weight η (<0 ⇒ keep bundle value)."))
    , d_etaT(initData(&d_etaT, -1.0, "etaT",
        "Override bundle dynamic-coherence weight η_t (<0 ⇒ keep bundle value)."))
    , d_localRank(initData(&d_localRank, -1, "localRank",
        "Override bundle local-POD rank r (<0 ⇒ keep bundle value)."))
    , d_geomEps(initData(&d_geomEps, 0.0, "geomEps",
        "FD step for the applyDJT geometric stiffness (∂J/∂q)ᵀf. "
        "<=0 disables geometric stiffness (first-order tangent only)."))
{
    d_kernelBundle.setPathType(sofa::core::objectmodel::PathType::DIRECTORY);
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::init()
{
    const unsigned n_in = this->fromModel->getSize();

    m_proj = sofa::component::kernel::loadPreImageProjectorFromBundle(
        d_kernelBundle.getValue());
    m_proj->setRegularization(d_eta.getValue(), d_etaT.getValue());
    m_proj->setLocalRank(d_localRank.getValue());

    msg_info(this) << "loaded pre-image bundle " << d_kernelBundle.getValue()
                   << "  kernel=" << m_proj->kernelName()
                   << "  3N="   << m_proj->nbDofs()
                   << "  T="    << m_proj->nbSnapshots()
                   << "  m="    << m_proj->nbModes()
                   << "  eta="  << m_proj->eta()
                   << "  etaT=" << m_proj->eta_t()
                   << "  r="    << m_proj->r()
                   << "  (reduced mstate dofs in=" << n_in << ")";

    if (m_proj->nbModes() != n_in)
        msg_error(this) << "Bundle has m=" << m_proj->nbModes()
                        << " modes but the reduced mstate requests " << n_in
                        << ". The pre-image head does not support rigid-mode "
                           "augmentation; m must equal the Vec1d dof count.";

    // Previous-step pre-image (η_t anchor + warm reference). Rest displacement
    // ≈ 0; the reduced state is expected to be initialised to q_rest=encode(0)
    // so the first decode lands at rest.
    m_u_prev.setZero(m_proj->nbDofs());

    // Skip Parent::init() (it would try to load modesPath); keep the
    // LinearMapping bookkeeping only, like KernelPCAMapping.
    sofa::component::mapping::linear::LinearMapping<TIn, TOut>::init();

    m_J_dirty = true;

    const auto& rot = this->d_rotation.getValue();
    if (rot[0] != 0.0 || rot[1] != 0.0 || rot[2] != 0.0)
        msg_warning(this) << "rotation Data is not wired for the pre-image mapping; ignored.";
}

template <class TIn, class TOut>
Eigen::VectorXd KernelPreImageMapping<TIn, TOut>::readLatent() const
{
    const InVecCoord q_vec =
        this->fromModel->read(core::vec_id::read_access::position)->getValue();
    Eigen::VectorXd q(static_cast<Eigen::Index>(q_vec.size()));
    for (Eigen::Index j = 0; j < q.size(); ++j)
        q(j) = q_vec[j][0];
    return q;
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::ensureJ()
{
    SCOPED_TIMER("ensureJ in KernelPreImageMapping");
    if (!m_J_dirty) return;
    const Eigen::VectorXd q = readLatent();
    const Eigen::VectorXd u_init = m_proj->uInit(q);
    sofa::helper::AdvancedTimer::stepBegin("ensureJ: compute J");
    m_J_cached = m_proj->jacobianLocal(q, u_init, m_u_prev);   // (3N, m)
    sofa::helper::AdvancedTimer::stepEnd("ensureJ: compute J");
    m_q_cached = q;
    m_J_dirty = false;
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::reset()
{
    m_u_prev.setZero(m_proj->nbDofs());
    m_J_dirty = true;
    Parent::reset();
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::apply(const core::MechanicalParams* /*mparams*/,
                                             Data<VecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    SCOPED_TIMER("apply in KernelPreImageMapping");

    helper::ReadAccessor<Data<InVecCoord>> in = dIn;
    const Eigen::Index m = static_cast<Eigen::Index>(in.size());
    Eigen::VectorXd q(m);
    for (Eigen::Index j = 0; j < m; ++j)
        q(j) = in[j][0];

    // EXACT decode: u = Ψ(q) (RBF fixed point), warm-referenced by u_prev for
    // the η_t coherence term. This replaces KernelPCAMapping's incremental
    // u_old + J·Δq — every step re-projects onto the manifold.
    const Eigen::VectorXd u_init = m_proj->uInit(q);
    const Eigen::VectorXd u = m_proj->solve(q, u_init, m_u_prev);   // displacement

    const Eigen::VectorXd& X0 = m_proj->X0();
    const Eigen::Index N = u.size() / 3;
    helper::WriteOnlyAccessor<Data<VecCoord>> out = dOut;
    out.resize(N);
    for (Eigen::Index i = 0; i < N; ++i)
        out[i] = Coord(X0(3 * i + 0) + u(3 * i + 0),
                       X0(3 * i + 1) + u(3 * i + 1),
                       X0(3 * i + 2) + u(3 * i + 2));

    m_u_prev = u;
    m_J_dirty = true;   // q changed; rebuild J(q) at next ensureJ
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::applyJ(const core::MechanicalParams* /*mparams*/,
                                              Data<VecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    SCOPED_TIMER("applyJ in KernelPreImageMapping");

    helper::WriteOnlyAccessor<Data<VecDeriv>> out = dOut;
    helper::ReadAccessor<Data<InVecDeriv>>    in  = dIn;

    sofa::helper::AdvancedTimer::stepBegin("applyJ: ensure J");
    ensureJ();
    sofa::helper::AdvancedTimer::stepEnd("applyJ: ensure J");

    sofa::helper::AdvancedTimer::stepBegin("applyJ: create Eigen vectors");
    const Eigen::Index N = static_cast<Eigen::Index>(m_J_cached.rows() / 3);
    const Eigen::Index m = static_cast<Eigen::Index>(in.size());
    sofa::helper::AdvancedTimer::stepEnd("applyJ: create Eigen vectors");

    sofa::helper::AdvancedTimer::stepBegin("applyJ: unpack input");
    Eigen::VectorXd dq(m);
    for (Eigen::Index j = 0; j < m; ++j)
        dq(j) = in[j][0];
    sofa::helper::AdvancedTimer::stepEnd("applyJ: unpack input");

    sofa::helper::AdvancedTimer::stepBegin("applyJ: compute du");
    const Eigen::VectorXd du = m_J_cached * dq;
    sofa::helper::AdvancedTimer::stepEnd("applyJ: compute du");

    sofa::helper::AdvancedTimer::stepBegin("applyJ: pack output");
    out.resize(N);
    for (Eigen::Index i = 0; i < N; ++i)
        out[i] = Deriv(du(3 * i + 0), du(3 * i + 1), du(3 * i + 2));
    sofa::helper::AdvancedTimer::stepEnd("applyJ: pack output");
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::applyJT(const core::MechanicalParams* /*mparams*/,
                                               Data<InVecDeriv>& dOut, const Data<VecDeriv>& dIn)
{
    SCOPED_TIMER("applyJT in KernelPreImageMapping");

    helper::WriteAccessor<Data<InVecDeriv>> out = dOut;
    helper::ReadAccessor<Data<VecDeriv>>    in  = dIn;

    ensureJ();
    const Eigen::Index N = static_cast<Eigen::Index>(in.size());
    Eigen::VectorXd f(3 * N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            f(3 * i + c) = in[i][c];

    const Eigen::VectorXd df = m_J_cached.transpose() * f;   // (m,) = J(q)ᵀ f
    for (Eigen::Index j = 0; j < df.size(); ++j)
        out[j][0] += df(j);
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams,
                                                core::MultiVecDerivId parentForceId,
                                                core::ConstMultiVecDerivId /*childForceId*/)
{
    // Geometric stiffness K_geo = (∂J/∂q)ᵀ f, added to the parent force as
    // K_geo·dq via a directional finite difference of the cached J:
    //   parentForce += kFactor · [ (J(q + ε·dq) − J(q)) / ε ]ᵀ · childForce.
    // Gated by geomEps so the first-order tangent (Stage A) can be measured
    // in isolation. KernelPCAMapping has K_geo ≡ 0 (and omits this entirely),
    // which is the "incomplete Jacobian" this head exists to complete.
    const double eps = d_geomEps.getValue();
    if (eps <= 0.0) return;

    SCOPED_TIMER("applyDJT in KernelPreImageMapping");
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

    const Eigen::VectorXd qp = m_q_cached + eps * dq;
    const Eigen::VectorXd u_init_p = m_proj->uInit(qp);
    const Eigen::MatrixXd Jp = m_proj->jacobianLocal(qp, u_init_p, m_u_prev);
    const Eigen::VectorXd dgeo = ((Jp - m_J_cached) / eps).transpose() * f;  // (m,)

    const double kfactor = static_cast<double>(mparams->kFactor());
    for (Eigen::Index j = 0; j < m; ++j)
        parentForce[j][0] += kfactor * dgeo(j);
}

template <class TIn, class TOut>
void KernelPreImageMapping<TIn, TOut>::applyJT(const core::ConstraintParams* /*cparams*/,
                                               Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn)
{
    SCOPED_TIMER("applyJT(constraint) in KernelPreImageMapping");

    InMatrixDeriv& out = *dOut.beginEdit();
    const MatrixDeriv& in = dIn.getValue();

    ensureJ();
    const Eigen::MatrixXd& J_u = m_J_cached;        // (3N, m)
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
