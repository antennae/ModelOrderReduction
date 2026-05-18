/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   AE counterpart of HyperReducedTriangleFEMForceFieldKPCA.inl
******************************************************************************/
#pragma once

#include <ModelOrderReduction/component/forcefield/HyperReducedTriangleFEMForceFieldAE.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/gl/template.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.h>

#include <Eigen/Core>


namespace sofa::component::forcefield
{

using sofa::component::loader::MatrixLoader;
using sofa::core::objectmodel::ComponentState;


namespace {
// Walk up the scene graph from `node` and return the first Vec1d
// MechanicalState found. Mirrors the helper in
// HyperReducedTetrahedronFEMForceFieldAE.inl — Link path resolution
// can fail when the scene was rewired by `modifyGraphSceneAE`.
inline sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>*
_find_vec1d_mstate_upwards_triangle(sofa::core::objectmodel::BaseContext* ctx)
{
    using sofa::defaulttype::Vec1Types;
    using sofa::core::behavior::MechanicalState;
    if (!ctx) return nullptr;
    auto* mstate = ctx->getMechanicalState();
    auto* typed  = dynamic_cast<MechanicalState<Vec1Types>*>(mstate);
    if (typed) return typed;
    auto* node = dynamic_cast<sofa::core::objectmodel::BaseNode*>(ctx);
    if (!node) return nullptr;
    for (auto* parent : node->getParents())
    {
        auto* found = _find_vec1d_mstate_upwards_triangle(parent->getContext());
        if (found) return found;
    }
    return nullptr;
}
} // anon


template <class DataTypes>
void HyperReducedTriangleFEMForceFieldAE<DataTypes>::init()
{
    TriangleFEMForceField<DataTypes>::init();
    this->initMOR(this->_indexedElements->size(), this->notMuted());

    m_qState = _find_vec1d_mstate_upwards_triangle(this->getContext());
    if (this->d_prepareECSW.getValue() && !m_qState)
    {
        msg_error(this) << "AE triangle force field requires a Vec1d MechanicalObject "
                          "in the parent chain (typically the AEMapping's "
                          "input mstate). None found.";
    }
}


template <class DataTypes>
void HyperReducedTriangleFEMForceFieldAE<DataTypes>::addForce(
    const core::MechanicalParams* /*mparams*/,
    DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /*v*/)
{
    VecDeriv& f1 = *f.beginEdit();
    const VecCoord& x1 = x.getValue();

    f1.resize(x1.size());

    // AE-specific: refresh the per-frame J(q_def) cache that updateGie consumes.
    if (this->d_prepareECSW.getValue() && m_qState)
    {
        const auto& q_pos = m_qState->read(core::vec_id::read_access::position)->getValue();
        const Eigen::Index nbTotal = static_cast<Eigen::Index>(q_pos.size());
        const Eigen::Index nbRigid = static_cast<Eigen::Index>(this->m_nbRigid);
        const Eigen::Index nbDef   = static_cast<Eigen::Index>(this->m_nbDef);
        if (nbTotal != nbRigid + nbDef)
        {
            msg_error(this) << "qState size " << nbTotal
                            << " != m_nbRigid (" << nbRigid << ") + m_nbDef (" << nbDef << ")";
        }
        else
        {
            Eigen::VectorXd q_def(nbDef);
            for (Eigen::Index j = 0; j < nbDef; ++j)
                q_def(j) = q_pos[nbRigid + j][0];
            this->prepareFrame(q_def);
        }
    }

    if (this->method == SMALL)
        this->accumulateForceSmall(f1, x1, true);
    else
        hyperReducedAccumulateForceLarge(f1, x1, true);

    f.endEdit();

    this->saveGieFile(this->_indexedElements->size());
}


template <class DataTypes>
void HyperReducedTriangleFEMForceFieldAE<DataTypes>::addDForce(
    const core::MechanicalParams* mparams,
    DataVecDeriv& df, const DataVecDeriv& dx)
{
    VecDeriv& df1 = *df.beginEdit();
    const VecDeriv& dx1 = dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    Real h = 1;
    df1.resize(dx1.size());

    if (this->method == SMALL)
        this->applyStiffnessSmall(df1, h, dx1, kFactor);
    else
        hyperReducedApplyStiffnessLarge(df1, h, dx1, kFactor);

    df.endEdit();
}


template <class DataTypes>
void HyperReducedTriangleFEMForceFieldAE<DataTypes>::hyperReducedAccumulateForceLarge(
    VecCoord& f, const VecCoord& p, bool implicit)
{
    typename VecElement::const_iterator it;
    unsigned int elementIndex(0);

    if (this->d_performECSW.getValue())
    {
        for (elementIndex = 0; elementIndex < this->m_RIDsize; ++elementIndex)
        {
            const Index elem = this->reducedIntegrationDomain(elementIndex);
            const Index a = (*_indexedElements)[elem][0];
            const Index b = (*_indexedElements)[elem][1];
            const Index c = (*_indexedElements)[elem][2];

            const Coord& pA = p[a];
            const Coord& pB = p[b];
            const Coord& pC = p[c];

            Transformation R_2_0(type::NOINIT), R_0_2(type::NOINIT);
            this->m_triangleUtils.computeRotationLarge(R_0_2, pA, pB, pC);

            const Coord deforme_b = R_0_2 * (pB - pA);
            const Coord deforme_c = R_0_2 * (pC - pA);

            Displacement Depl(type::NOINIT);
            this->m_triangleUtils.computeDisplacementLarge(Depl, R_0_2, _rotatedInitialElements[elem], pA, pB, pC);

            StrainDisplacement J(type::NOINIT);
            try
            {
                this->m_triangleUtils.computeStrainDisplacementLocal(J, deforme_b, deforme_c);
            }
            catch (const std::exception& e)
            {
                msg_error() << e.what();
                sofa::core::objectmodel::BaseObject::d_componentState.setValue(ComponentState::Invalid);
                break;
            }

            type::Vec<3, Real> strain(type::NOINIT);
            this->m_triangleUtils.computeStrain(strain, J, Depl, false);

            type::Vec<3, Real> stress(type::NOINIT);
            this->m_triangleUtils.computeStress(stress, _materialsStiffnesses[elem], strain, false);

            Displacement F(type::NOINIT);
            this->m_triangleUtils.computeForceLarge(F, J, stress);

            R_2_0.transpose(R_0_2);
            std::vector<Deriv> contrib;
            std::vector<unsigned int> indexList;
            contrib.resize(3);
            indexList.resize(3);
            contrib[0] = R_2_0 * Coord(F[0], F[1], 0);
            contrib[1] = R_2_0 * Coord(F[2], F[3], 0);
            contrib[2] = R_2_0 * Coord(F[4], F[5], 0);
            indexList[0] = a;
            indexList[1] = b;
            indexList[2] = c;
            for (auto i : {0, 1, 2})
                f[indexList[i]] += this->weights(elem) * contrib[i];

            if (implicit)
            {
                _strainDisplacements[elem] = J;
                _rotations[elem] = R_2_0;
            }
        }
    }
    else
    {
        for (it = _indexedElements->begin(); it != _indexedElements->end(); ++it, ++elementIndex)
        {
            const Index a = (*_indexedElements)[elementIndex][0];
            const Index b = (*_indexedElements)[elementIndex][1];
            const Index c = (*_indexedElements)[elementIndex][2];

            const Coord& pA = p[a];
            const Coord& pB = p[b];
            const Coord& pC = p[c];

            Transformation R_2_0(type::NOINIT), R_0_2(type::NOINIT);
            this->m_triangleUtils.computeRotationLarge(R_0_2, pA, pB, pC);

            const Coord deforme_b = R_0_2 * (pB - pA);
            const Coord deforme_c = R_0_2 * (pC - pA);

            Displacement Depl(type::NOINIT);
            this->m_triangleUtils.computeDisplacementLarge(Depl, R_0_2, _rotatedInitialElements[elementIndex], pA, pB, pC);

            StrainDisplacement J(type::NOINIT);
            try
            {
                this->m_triangleUtils.computeStrainDisplacementLocal(J, deforme_b, deforme_c);
            }
            catch (const std::exception& e)
            {
                msg_error() << e.what();
                sofa::core::objectmodel::BaseObject::d_componentState.setValue(ComponentState::Invalid);
                break;
            }

            type::Vec<3, Real> strain(type::NOINIT);
            this->m_triangleUtils.computeStrain(strain, J, Depl, false);

            type::Vec<3, Real> stress(type::NOINIT);
            this->m_triangleUtils.computeStress(stress, _materialsStiffnesses[elementIndex], strain, false);

            Displacement F(type::NOINIT);
            this->m_triangleUtils.computeForceLarge(F, J, stress);

            R_2_0.transpose(R_0_2);
            std::vector<Deriv> contrib;
            std::vector<unsigned int> indexList;
            contrib.resize(3);
            indexList.resize(3);
            contrib[0] = R_2_0 * Coord(F[0], F[1], 0);
            contrib[1] = R_2_0 * Coord(F[2], F[3], 0);
            contrib[2] = R_2_0 * Coord(F[4], F[5], 0);
            indexList[0] = a;
            indexList[1] = b;
            indexList[2] = c;
            for (auto i : {0, 1, 2})
                f[indexList[i]] += contrib[i];

            this->template updateGie<DataTypes>(indexList, contrib, elementIndex);

            if (implicit)
            {
                _strainDisplacements[elementIndex] = J;
                _rotations[elementIndex] = R_2_0;
            }
        }
    }
}


template <class DataTypes>
void HyperReducedTriangleFEMForceFieldAE<DataTypes>::hyperReducedApplyStiffnessLarge(
    VecCoord& v, Real h, const VecCoord& x, const SReal& kFactor)
{
    unsigned int i;
    typename VecElement::const_iterator it, it0;

    it0 = _indexedElements->begin();
    unsigned int nbElementsConsidered;
    const bool performECSW = this->d_performECSW.getValue();
    if (!performECSW)
        nbElementsConsidered = _indexedElements->size();
    else
        nbElementsConsidered = this->m_RIDsize;

    for (unsigned int numElem = 0; numElem < nbElementsConsidered; ++numElem)
    {
        if (!performECSW)
            i = numElem;
        else
            i = this->reducedIntegrationDomain(numElem);
        it = it0 + i;

        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        Transformation R_0_2(type::NOINIT);
        R_0_2.transpose(_rotations[i]);

        Displacement dX(type::NOINIT);

        Coord x_2 = R_0_2 * x[a];
        dX[0] = x_2[0];
        dX[1] = x_2[1];

        x_2 = R_0_2 * x[b];
        dX[2] = x_2[0];
        dX[3] = x_2[1];

        x_2 = R_0_2 * x[c];
        dX[4] = x_2[0];
        dX[5] = x_2[1];

        type::Vec<3, Real> strain(type::NOINIT);
        this->m_triangleUtils.computeStrain(strain, _strainDisplacements[i], dX, false);

        type::Vec<3, Real> stress(type::NOINIT);
        this->m_triangleUtils.computeStress(stress, _materialsStiffnesses[i], strain, false);

        Displacement F(type::NOINIT);
        this->m_triangleUtils.computeForceLarge(F, _strainDisplacements[i], stress);

        if (!performECSW)
        {
            v[a] += (_rotations[i] * Coord(-h * F[0], -h * F[1], 0)) * kFactor;
            v[b] += (_rotations[i] * Coord(-h * F[2], -h * F[3], 0)) * kFactor;
            v[c] += (_rotations[i] * Coord(-h * F[4], -h * F[5], 0)) * kFactor;
        }
        else
        {
            v[a] += this->weights(i) * (_rotations[i] * Coord(-h * F[0], -h * F[1], 0)) * kFactor;
            v[b] += this->weights(i) * (_rotations[i] * Coord(-h * F[2], -h * F[3], 0)) * kFactor;
            v[c] += this->weights(i) * (_rotations[i] * Coord(-h * F[4], -h * F[5], 0)) * kFactor;
        }
    }
}


template <class DataTypes>
void HyperReducedTriangleFEMForceFieldAE<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowForceFields())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    typename VecElement::const_iterator it, it0;
    it0 = _indexedElements->begin();
    for (unsigned i = 0; i < this->m_RIDsize; ++i)
    {
        it = it0 + this->reducedIntegrationDomain(i);
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        glColor4f(0, 1, 0, 1);
        gl::glVertexT(x[a]);
        glColor4f(0, 0.5, 0.5, 1);
        gl::glVertexT(x[b]);
        glColor4f(0, 0, 1, 1);
        gl::glVertexT(x[c]);
    }
    glEnd();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif
}


template <class DataTypes>
void HyperReducedTriangleFEMForceFieldAE<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    StiffnessMatrix JKJt, RJKJtRt;
    sofa::type::Mat<3, 3, Real> localMatrix(type::NOINIT);

    constexpr auto S = DataTypes::deriv_total_size;
    constexpr auto N = Element::size();

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                    .withRespectToPositionsIn(this->mstate);

    sofa::Size triangleId = 0;

    typename VecElement::const_iterator it;
    auto it0 = _indexedElements->begin();
    int nbElementsConsidered;

    const bool performECSW = this->d_performECSW.getValue();
    if (!performECSW)
        nbElementsConsidered = _indexedElements->size();
    else
        nbElementsConsidered = this->m_RIDsize;

    for (unsigned int numElem = 0; numElem < nbElementsConsidered; ++numElem)
    {
        if (!performECSW)
            triangleId = numElem;
        else
            triangleId = this->reducedIntegrationDomain(numElem);
        it = it0 + triangleId;

        this->computeElementStiffnessMatrix(JKJt, RJKJtRt, _materialsStiffnesses[triangleId], _strainDisplacements[triangleId], _rotations[triangleId]);

        for (sofa::Index n1 = 0; n1 < N; n1++)
        {
            for (sofa::Index n2 = 0; n2 < N; n2++)
            {
                RJKJtRt.getsub(S * n1, S * n2, localMatrix);
                if (!performECSW)
                    dfdx((*it)[n1] * S, (*it)[n2] * S) += -localMatrix;
                else
                    dfdx((*it)[n1] * S, (*it)[n2] * S) += -localMatrix * this->weights(triangleId);
            }
        }
    }
}


} // namespace sofa::component::forcefield
