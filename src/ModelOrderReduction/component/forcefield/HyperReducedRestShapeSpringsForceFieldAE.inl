/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   AE counterpart of HyperReducedRestShapeSpringsForceFieldKPCA.inl
******************************************************************************/
#pragma once

#include <ModelOrderReduction/component/forcefield/HyperReducedRestShapeSpringsForceFieldAE.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/helper/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

#include <Eigen/Core>

#include <assert.h>
#include <iostream>

namespace
{
    template <typename...> using void_t = void;
    template <typename, template <typename> class, typename = void_t<>>
    struct detect : std::false_type {};
    template <typename T, template <typename> class Op>
    struct detect<T, Op, void_t<Op<T>>> : std::true_type {};
    template <typename T>
    using isRigid_t = decltype(std::declval<typename T::Coord>().getOrientation());
    template <typename T>
    using isRigidType = detect<T, isRigid_t>;
}

namespace sofa::component::solidmechanics::spring
{

using helper::WriteAccessor;
using helper::ReadAccessor;
using core::behavior::BaseMechanicalState;
using core::behavior::MultiMatrixAccessor;
using core::behavior::ForceField;
using linearalgebra::BaseMatrix;
using core::VecCoordId;
using core::MechanicalParams;
using type::Vec3;
using type::Vec4f;
using type::vector;
using core::visual::VisualParams;

namespace {
inline sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>*
_find_vec1d_mstate_upwards(sofa::core::objectmodel::BaseContext* ctx)
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
        auto* found = _find_vec1d_mstate_upwards(parent->getContext());
        if (found) return found;
    }
    return nullptr;
}
} // anon

template<class DataTypes>
HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::HyperReducedRestShapeSpringsForceFieldAE()
{
}

template<class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::parse(core::objectmodel::BaseObjectDescription *arg)
{
    const char* attr = arg->getAttribute("external_rest_shape");
    if( attr != nullptr && attr[0] != '@')
        msg_error() << "HyperReducedRestShapeSpringsForceFieldAE: 'external_rest_shape' must be a Link prefixed with '@'.";
    Inherit::parse(arg);
}

template<class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::bwdInit()
{
    ForceField<DataTypes>::init();

    if (d_stiffness.getValue().empty())
    {
        msg_info() << "No stiffness is defined, assuming equal stiffness on each node, k = 100.0 ";
        VecReal stiffs;
        stiffs.push_back(100.0);
        d_stiffness.setValue(stiffs);
    }

    if (l_restMState.get() == NULL)
    {
        useRestMState = false;
        msg_info() << "no external rest shape used";
        if(!l_restMState.empty())
            msg_warning() << "external_rest_shape in node " << this->getContext()->getName() << " not found";
    }
    else
    {
        msg_info() << "external rest shape used";
        useRestMState = true;
    }

    recomputeIndices();

    BaseMechanicalState* state = this->getContext()->getMechanicalState();
    if(!state)
        msg_warning() << "MechanicalState of the current context returns null pointer";
    else
    {
        assert(state);
        matS.resize(state->getMatrixSize(), state->getMatrixSize());
    }
    lastUpdatedStep = -1.0;
    this->initMOR(d_points.getValue().size(), notMuted());

    m_qState = _find_vec1d_mstate_upwards(this->getContext());
    if (this->d_prepareECSW.getValue() && !m_qState)
    {
        msg_error(this) << "AE rest-shape springs require a Vec1d MechanicalObject "
                          "in the parent chain (typically the AEMapping's "
                          "input mstate). None found.";
    }
}

template<class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::reinit()
{
    if (d_stiffness.getValue().empty())
    {
        msg_info() << "No stiffness is defined, assuming equal stiffness on each node, k = 100.0 ";
        VecReal stiffs;
        stiffs.push_back(100.0);
        d_stiffness.setValue(stiffs);
    }
}

template<class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::recomputeIndices()
{
    m_indices.clear();
    m_ext_indices.clear();

    for (unsigned int i = 0; i < d_points.getValue().size(); i++)
        m_indices.push_back(d_points.getValue()[i]);
    for (unsigned int i = 0; i < d_external_points.getValue().size(); i++)
        m_ext_indices.push_back(d_external_points.getValue()[i]);

    if (m_indices.empty())
        for (unsigned int i = 0; i < (unsigned)this->mstate->getSize(); i++)
            m_indices.push_back(i);

    if (m_ext_indices.empty())
    {
        if (useRestMState)
            for (unsigned int i = 0; i < getExtPosition()->getValue().size(); i++)
                m_ext_indices.push_back(i);
        else
            for (unsigned int i = 0; i < m_indices.size(); i++)
                m_ext_indices.push_back(m_indices[i]);
    }

    if (!checkOutOfBoundsIndices())
    {
        msg_error() << "The dimension of the source and the targeted points are different ";
        m_indices.clear();
    }
    else
        msg_info() << "Indices successfully checked";
}

template<class DataTypes>
bool HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::checkOutOfBoundsIndices()
{
    if (!RestShapeSpringsForceField<DataTypes>::checkOutOfBoundsIndices(m_indices, this->mstate->getSize()))
    {
        msg_error() << "Out of Bounds m_indices detected. ForceField is not activated.";
        return false;
    }
    if (!RestShapeSpringsForceField<DataTypes>::checkOutOfBoundsIndices(m_ext_indices, getExtPosition()->getValue().size()))
    {
        msg_error() << "Out of Bounds m_ext_indices detected. ForceField is not activated.";
        return false;
    }
    if (m_indices.size() != m_ext_indices.size())
    {
        msg_error() << "Dimensions of the source and the targeted points are different. ForceField is not activated.";
        return false;
    }
    return true;
}

template<class DataTypes>
const typename HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::DataVecCoord*
HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::getExtPosition() const
{
    return (useRestMState ? l_restMState->read(sofa::core::vec_id::read_access::position)
                          : this->mstate->read(sofa::core::vec_id::read_access::restPosition));
}

template<class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::addForce(const MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);
    WriteAccessor< DataVecDeriv > f1 = f;
    ReadAccessor< DataVecCoord > p1 = x;
    ReadAccessor< DataVecCoord > p0 = *this->getExtPosition();

    f1.resize(p1.size());

    if (d_recompute_indices.getValue())
        this->recomputeIndices();

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

    unsigned int i;
    const VecReal &k = d_stiffness.getValue();
    if ( k.size()!= m_indices.size() )
    {
        const Real k0 = k[0];
        unsigned int nbElementsConsidered;
        if (!d_performECSW.getValue())
            nbElementsConsidered = m_indices.size();
        else
        {
            if (m_RIDsize != 0)
                nbElementsConsidered = m_RIDsize;
            else
            {
                msg_warning() << "RID is empty!!! Taking all the elements...";
                nbElementsConsidered = m_indices.size();
            }
        }
        for (unsigned int point = 0 ; point<nbElementsConsidered ;++point)
        {
            if (!d_performECSW.getValue())
                i = point;
            else
                i = reducedIntegrationDomain(point);

            const unsigned int index = m_indices[i];
            unsigned int ext_index = m_indices[i];
            if(useRestMState)
                ext_index= m_ext_indices[i];

            Deriv dx = p1[index] - p0[ext_index];
            std::vector<Deriv> contrib;
            std::vector<unsigned int> indexList;
            contrib.resize(1);
            indexList.resize(1);
            contrib[0] = -dx * k0;
            indexList[0] = index;
            if (!d_performECSW.getValue())
                f1[index] += contrib[0];
            else
                f1[index] += weights(i) * contrib[0];

            this->template updateGie<DataTypes>(indexList, contrib, i);
        }
    }
    else
    {
        unsigned int nbElementsConsidered;
        if (!d_performECSW.getValue())
            nbElementsConsidered = m_indices.size();
        else
        {
            if (m_RIDsize != 0)
                nbElementsConsidered = m_RIDsize;
            else
            {
                nbElementsConsidered = m_indices.size();
                msg_warning("RID is empty! Taking all the elements...");
            }
        }
        for (unsigned int point = 0 ; point<nbElementsConsidered ;++point)
        {
            if (!d_performECSW.getValue())
                i = point;
            else
                i = reducedIntegrationDomain(point);

            const unsigned int index = m_indices[i];
            unsigned int ext_index = m_indices[i];
            if(useRestMState)
                ext_index= m_ext_indices[i];

            Deriv dx = p1[index] - p0[ext_index];
            std::vector<Deriv> contrib;
            std::vector<unsigned int> indexList;
            contrib.resize(1);
            indexList.resize(1);
            contrib[0] = -dx * k[i];
            indexList[0] = index;
            if (!d_performECSW.getValue())
                f1[index] += contrib[0];
            else
                f1[index] += weights(i) * contrib[0];
            this->template updateGie<DataTypes>(indexList, contrib, i);
        }
    }
    this->saveGieFile(m_indices.size());
}

template<class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::addDForce(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    WriteAccessor< DataVecDeriv > df1 = df;
    ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    const VecReal &k = d_stiffness.getValue();
    if (k.size() != m_indices.size())
    {
        const Real k0 = k[0];
        if (d_performECSW.getValue()){
            for(unsigned int i = 0 ; i<m_RIDsize ;++i)
                df1[m_indices[reducedIntegrationDomain(i)]] -= weights(reducedIntegrationDomain(i)) * dx1[m_indices[reducedIntegrationDomain(i)]] * k0 * kFactor;
        }
        else
        {
            for (unsigned int i=0; i<m_indices.size(); i++)
                df1[m_indices[i]] -= dx1[m_indices[i]] * k0 * kFactor;
        }
    }
    else
    {
        if (d_performECSW.getValue()){
            for(unsigned int i = 0 ; i<m_RIDsize ;++i)
                df1[m_indices[reducedIntegrationDomain[i]]] -= weights(reducedIntegrationDomain(i)) * dx1[m_indices[reducedIntegrationDomain(i)]] * k[reducedIntegrationDomain(i)] * kFactor;
        }
        else
        {
            for (unsigned int i=0; i<m_indices.size(); i++)
                df1[m_indices[i]] -= dx1[m_indices[i]] * k[i] * kFactor;
        }
    }
}

template<class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::draw(const VisualParams *vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !d_drawSpring.getValue())
        return;
    if(DataTypes::spatial_dimensions > 3)
    {
        msg_error() << "Draw function not implemented for this DataType";
        return;
    }

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->setLightingEnabled(false);

    ReadAccessor< DataVecCoord > p0 = *this->getExtPosition();
    ReadAccessor< DataVecCoord > p  = this->mstate->read(sofa::core::vec_id::read_access::position);

    const VecIndex& indices = m_indices;
    const VecIndex& ext_indices = (useRestMState ? m_ext_indices : m_indices);

    vector<type::Vec3> vertices;
    if (d_performECSW.getValue()){
        for (unsigned int i=0; i<m_RIDsize; i++)
        {
            const unsigned int index = indices[reducedIntegrationDomain(i)];
            const unsigned int ext_index = ext_indices[reducedIntegrationDomain(i)];

            type::Vec3 v0(0.0, 0.0, 0.0);
            type::Vec3 v1(0.0, 0.0, 0.0);
            for(unsigned int j=0 ; j<DataTypes::spatial_dimensions ; j++)
            {
                v0[j] = p[index][j];
                v1[j] = p0[ext_index][j];
            }
            vertices.push_back(v0);
            vertices.push_back(v1);
        }
    }
    else
    {
        for (unsigned int i=0; i<indices.size(); i++)
        {
            const unsigned int index = indices[i];
            const unsigned int ext_index = ext_indices[i];

            type::Vec3 v0(0.0, 0.0, 0.0);
            type::Vec3 v1(0.0, 0.0, 0.0);
            for(unsigned int j=0 ; j<DataTypes::spatial_dimensions ; j++)
            {
                v0[j] = p[index][j];
                v1[j] = p0[ext_index][j];
            }
            vertices.push_back(v0);
            vertices.push_back(v1);
        }
    }
    vparams->drawTool()->drawLines(vertices, 5, sofa::type::RGBAColor(d_springColor.getValue()));
    vparams->drawTool()->restoreLastState();
}

template <class DataTypes>
void HyperReducedRestShapeSpringsForceFieldAE<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();
    const auto activeDirections = d_activeDirections.getValue();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions;
    constexpr sofa::Size total_size = Deriv::total_size;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);
    unsigned int nbIndicesConsidered;
    sofa::Index curIndex;

    if (d_performECSW.getValue())
        nbIndicesConsidered = m_RIDsize;
    else
        nbIndicesConsidered = m_indices.size();

    for (sofa::Index index = 0; index < nbIndicesConsidered; index++)
    {
        if (!d_performECSW.getValue())
            curIndex = m_indices[index];
        else
            curIndex = m_indices[reducedIntegrationDomain(index)];

        const auto vt = -k[(curIndex < k.size()) * curIndex];
        for(sofa::Index i = 0; i < space_size; i++)
        {
            if (!d_performECSW.getValue())
                dfdx(total_size * curIndex + i, total_size * curIndex + i) += vt;
            else
                dfdx(total_size * curIndex + i, total_size * curIndex + i) += vt * weights(reducedIntegrationDomain(index));
        }

        if constexpr (isRigidType<DataTypes>())
        {
            const auto vr = -k_a[(curIndex < k_a.size()) * curIndex];
            for (sofa::Size i = space_size; i < total_size; ++i)
            {
                if (activeDirections[i])
                {
                    if (!d_performECSW.getValue())
                        dfdx(total_size * index + i, total_size * index + i) += vr;
                    else
                        dfdx(total_size * index + i, total_size * index + i) += vr * weights(reducedIntegrationDomain(index));
                }
            }
        }
    }
}

} // namespace sofa::component::solidmechanics::spring
