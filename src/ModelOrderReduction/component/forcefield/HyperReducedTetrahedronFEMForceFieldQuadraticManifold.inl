/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   quadratic-manifold counterpart of HyperReducedTetrahedronFEMForceFieldKPCA.inl
******************************************************************************/
#pragma once

#include "sofa/core/behavior/BaseLocalForceFieldMatrix.h"
#include <ModelOrderReduction/component/forcefield/HyperReducedTetrahedronFEMForceFieldQuadraticManifold.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/container/grid/GridTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/decompose.h>
#include <sofa/gl/template.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <set>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/system/thread/CTime.h>

#include <Eigen/Core>


namespace sofa::component::forcefield
{

using sofa::component::loader::MatrixLoader;
using sofa::core::objectmodel::ComponentState;


namespace {
// Walk up the scene graph from `node` and return the first Vec1d
// MechanicalState whose size equals `expectedSize`. Returns nullptr if
// none. Used at init() time because Link path resolution can fail when the
// scene was rewired by `modifyGraphSceneQuadraticManifold` between addObject and SOFA's
// init pass.
//
// The size filter is essential on Rigidify + articulated scenes: the
// reduced node is cross-parented through BOTH the deformable branch (which
// carries the size-(nbRigid+nbDef) modal MO we want) AND the rigid/servo
// branch (whose Articulation angle states are also Vec1d, but size 1). A
// size-blind walk returns whichever it reaches first — typically a servo
// angle — yielding `qState size 1 != nbRigid + nbDef` and all-zero Gie.
inline sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>*
_find_vec1d_mstate_upwards(sofa::core::objectmodel::BaseContext* ctx,
                           std::size_t expectedSize)
{
    using sofa::defaulttype::Vec1Types;
    using sofa::core::behavior::MechanicalState;
    if (!ctx) return nullptr;
    auto* mstate = ctx->getMechanicalState();
    auto* typed  = dynamic_cast<MechanicalState<Vec1Types>*>(mstate);
    if (typed && static_cast<std::size_t>(typed->getSize()) == expectedSize)
        return typed;
    // Recurse into parents (other Vec1d states of the wrong size are skipped).
    auto* node = dynamic_cast<sofa::core::objectmodel::BaseNode*>(ctx);
    if (!node) return nullptr;
    for (auto* parent : node->getParents())
    {
        auto* found = _find_vec1d_mstate_upwards(parent->getContext(), expectedSize);
        if (found) return found;
    }
    return nullptr;
}
} // anon

template<class DataTypes>
HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::HyperReducedTetrahedronFEMForceFieldQuadraticManifold()
{
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::accumulateForceSmall( Vector& f, const Vector & p, VecElement::const_iterator elementIt, Index elementIndex )
{
    TetrahedronFEMForceField<DataTypes>::accumulateForceSmall( f, p, elementIt, elementIndex );
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::applyStiffnessSmall( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact )
{
    TetrahedronFEMForceField<DataTypes>::applyStiffnessSmall( f, x, i, a, b, c, d, fact );
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::accumulateForceLarge( Vector& f, const Vector & p, VecElement::const_iterator elementIt, Index elementIndex )
{
    Element index = *elementIt;

    Transformation R_0_2;
    this->computeRotationLarge( R_0_2, p, index[0],index[1],index[2]);

    rotations[elementIndex].transpose(R_0_2);

    type::fixed_array<Coord,4> deforme;
    for(int i=0; i<4; ++i)
        deforme[i] = R_0_2*p[index[i]];

    deforme[1][0] -= deforme[0][0];
    deforme[2][0] -= deforme[0][0];
    deforme[2][1] -= deforme[0][1];
    deforme[3] -= deforme[0];

    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] = _rotatedInitialElements[elementIndex][1][0] - deforme[1][0];
    D[4] = 0;
    D[5] = 0;
    D[6] = _rotatedInitialElements[elementIndex][2][0] - deforme[2][0];
    D[7] = _rotatedInitialElements[elementIndex][2][1] - deforme[2][1];
    D[8] = 0;
    D[9] = _rotatedInitialElements[elementIndex][3][0] - deforme[3][0];
    D[10] = _rotatedInitialElements[elementIndex][3][1] - deforme[3][1];
    D[11] =_rotatedInitialElements[elementIndex][3][2] - deforme[3][2];

    Displacement F;
    if (this->d_updateStiffnessMatrix.getValue())
    {
        strainDisplacements[elementIndex][0][0]   = ( - deforme[2][1]*deforme[3][2] );
        strainDisplacements[elementIndex][1][1] = ( deforme[2][0]*deforme[3][2] - deforme[1][0]*deforme[3][2] );
        strainDisplacements[elementIndex][2][2]   = ( deforme[2][1]*deforme[3][0] - deforme[2][0]*deforme[3][1] + deforme[1][0]*deforme[3][1] - deforme[1][0]*deforme[2][1] );

        strainDisplacements[elementIndex][3][0]   = ( deforme[2][1]*deforme[3][2] );
        strainDisplacements[elementIndex][4][1]  = ( - deforme[2][0]*deforme[3][2] );
        strainDisplacements[elementIndex][5][2]   = ( - deforme[2][1]*deforme[3][0] + deforme[2][0]*deforme[3][1] );

        strainDisplacements[elementIndex][7][1]  = ( deforme[1][0]*deforme[3][2] );
        strainDisplacements[elementIndex][8][2]   = ( - deforme[1][0]*deforme[3][1] );

        strainDisplacements[elementIndex][11][2] = ( deforme[1][0]*deforme[2][1] );
    }

    if(!this->d_assembling.getValue())
    {
        this->computeForce( F, D, _plasticStrains[elementIndex], materialsStiffnesses[elementIndex], strainDisplacements[elementIndex] );
        std::vector<Deriv> contrib;
        std::vector<unsigned int> indexList;
        contrib.resize(4);
        indexList.resize(4);
        for(int i=0; i<12; i+=3){
            contrib[i/3] = rotations[elementIndex] * Deriv( F[i], F[i+1],  F[i+2] );
            indexList[i/3] = index[i/3];
            if (!d_performECSW.getValue())
                f[indexList[i/3]] +=  contrib[i/3];
            else
                f[indexList[i/3]] +=  weights(elementIndex)*contrib[i/3];
        }
        this->template updateGie<DataTypes>(indexList, contrib, elementIndex);
    }
    else if(this->d_plasticMaxThreshold.getValue() <= 0 )
    {
        strainDisplacements[elementIndex][6][0] = 0;
        strainDisplacements[elementIndex][9][0] = 0;
        strainDisplacements[elementIndex][10][1] = 0;

        StiffnessMatrix RJKJt, RJKJtRt;
        this->computeStiffnessMatrix(RJKJt,RJKJtRt,materialsStiffnesses[elementIndex], strainDisplacements[elementIndex],rotations[elementIndex]);

        if(elementIndex==0)
        {
            for(unsigned int i=0; i<this->_stiffnesses.size(); ++i)
                this->_stiffnesses[i].resize(0);
        }

        for(int i=0; i<12; ++i)
        {
            index_type row = index[i/3]*3+i%3;
            for(int j=0; j<12; ++j)
            {
                index_type col = index[j/3]*3+j%3;

                typename CompressedValue::iterator result = this->_stiffnesses[row].end();
                for (auto it = this->_stiffnesses[row].begin();
                    it != this->_stiffnesses[row].end() && result == this->_stiffnesses[row].end(); ++it)
                {
                    if( (*it).first == col )
                        result = it;
                }

                if( result==this->_stiffnesses[row].end() )
                    this->_stiffnesses[row].push_back( Col_Value(col,RJKJtRt[i][j] )  );
                else
                    (*result).second += RJKJtRt[i][j];
            }
        }
        F = RJKJt*D;

        for(int i=0; i<12; i+=3)
            f[index[i/3]] += Deriv( F[i], F[i+1],  F[i+2] );
    }
    else
    {
        msg_warning(this) << "TODO(HyperReducedTetrahedronFEMForceFieldQuadraticManifold): support for assembling system matrix when using plasticity.";
    }
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::accumulateForcePolar( Vector& f, const Vector & p, VecElement::const_iterator elementIt, Index elementIndex )
{
    TetrahedronFEMForceField<DataTypes>::accumulateForcePolar( f, p, elementIt, elementIndex );
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::accumulateForceSVD( Vector& f, const Vector & p, VecElement::const_iterator elementIt, Index elementIndex )
{
   TetrahedronFEMForceField<DataTypes>::accumulateForceSVD( f, p, elementIt, elementIndex );
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::applyStiffnessCorotational( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact )
{
    Displacement X;

    X[0]  = rotations[i][0][0] * x[a][0] + rotations[i][1][0] * x[a][1] + rotations[i][2][0] * x[a][2];
    X[1]  = rotations[i][0][1] * x[a][0] + rotations[i][1][1] * x[a][1] + rotations[i][2][1] * x[a][2];
    X[2]  = rotations[i][0][2] * x[a][0] + rotations[i][1][2] * x[a][1] + rotations[i][2][2] * x[a][2];

    X[3]  = rotations[i][0][0] * x[b][0] + rotations[i][1][0] * x[b][1] + rotations[i][2][0] * x[b][2];
    X[4]  = rotations[i][0][1] * x[b][0] + rotations[i][1][1] * x[b][1] + rotations[i][2][1] * x[b][2];
    X[5]  = rotations[i][0][2] * x[b][0] + rotations[i][1][2] * x[b][1] + rotations[i][2][2] * x[b][2];

    X[6]  = rotations[i][0][0] * x[c][0] + rotations[i][1][0] * x[c][1] + rotations[i][2][0] * x[c][2];
    X[7]  = rotations[i][0][1] * x[c][0] + rotations[i][1][1] * x[c][1] + rotations[i][2][1] * x[c][2];
    X[8]  = rotations[i][0][2] * x[c][0] + rotations[i][1][2] * x[c][1] + rotations[i][2][2] * x[c][2];

    X[9]  = rotations[i][0][0] * x[d][0] + rotations[i][1][0] * x[d][1] + rotations[i][2][0] * x[d][2];
    X[10] = rotations[i][0][1] * x[d][0] + rotations[i][1][1] * x[d][1] + rotations[i][2][1] * x[d][2];
    X[11] = rotations[i][0][2] * x[d][0] + rotations[i][1][2] * x[d][1] + rotations[i][2][2] * x[d][2];

    Displacement F;
    this->computeForce( F, X, materialsStiffnesses[i], strainDisplacements[i], fact );

    if (d_performECSW.getValue()){
        f[a][0] -=  weights(i)*(rotations[i][0][0] *  F[0] +  rotations[i][0][1] * F[1]  + rotations[i][0][2] * F[2]);
        f[a][1] -=  weights(i)*(rotations[i][1][0] *  F[0] +  rotations[i][1][1] * F[1]  + rotations[i][1][2] * F[2]);
        f[a][2] -=  weights(i)*(rotations[i][2][0] *  F[0] +  rotations[i][2][1] * F[1]  + rotations[i][2][2] * F[2]);

        f[b][0] -=  weights(i)*(rotations[i][0][0] *  F[3] +  rotations[i][0][1] * F[4]  + rotations[i][0][2] * F[5]);
        f[b][1] -=  weights(i)*(rotations[i][1][0] *  F[3] +  rotations[i][1][1] * F[4]  + rotations[i][1][2] * F[5]);
        f[b][2] -=  weights(i)*(rotations[i][2][0] *  F[3] +  rotations[i][2][1] * F[4]  + rotations[i][2][2] * F[5]);

        f[c][0] -=  weights(i)*(rotations[i][0][0] *  F[6] +  rotations[i][0][1] * F[7]  + rotations[i][0][2] * F[8]);
        f[c][1] -=  weights(i)*(rotations[i][1][0] *  F[6] +  rotations[i][1][1] * F[7]  + rotations[i][1][2] * F[8]);
        f[c][2] -=  weights(i)*(rotations[i][2][0] *  F[6] +  rotations[i][2][1] * F[7]  + rotations[i][2][2] * F[8]);

        f[d][0] -=  weights(i)*(rotations[i][0][0] *  F[9] +  rotations[i][0][1] * F[10] + rotations[i][0][2] * F[11]);
        f[d][1] -=  weights(i)*(rotations[i][1][0] *  F[9] +  rotations[i][1][1] * F[10] + rotations[i][1][2] * F[11]);
        f[d][2] -=  weights(i)*(rotations[i][2][0] *  F[9] +  rotations[i][2][1] * F[10] + rotations[i][2][2] * F[11]);
    }
    else
    {
        f[a][0] -= rotations[i][0][0] *  F[0] +  rotations[i][0][1] * F[1]  + rotations[i][0][2] * F[2];
        f[a][1] -= rotations[i][1][0] *  F[0] +  rotations[i][1][1] * F[1]  + rotations[i][1][2] * F[2];
        f[a][2] -= rotations[i][2][0] *  F[0] +  rotations[i][2][1] * F[1]  + rotations[i][2][2] * F[2];

        f[b][0] -= rotations[i][0][0] *  F[3] +  rotations[i][0][1] * F[4]  + rotations[i][0][2] * F[5];
        f[b][1] -= rotations[i][1][0] *  F[3] +  rotations[i][1][1] * F[4]  + rotations[i][1][2] * F[5];
        f[b][2] -= rotations[i][2][0] *  F[3] +  rotations[i][2][1] * F[4]  + rotations[i][2][2] * F[5];

        f[c][0] -= rotations[i][0][0] *  F[6] +  rotations[i][0][1] * F[7]  + rotations[i][0][2] * F[8];
        f[c][1] -= rotations[i][1][0] *  F[6] +  rotations[i][1][1] * F[7]  + rotations[i][1][2] * F[8];
        f[c][2] -= rotations[i][2][0] *  F[6] +  rotations[i][2][1] * F[7]  + rotations[i][2][2] * F[8];

        f[d][0] -= rotations[i][0][0] *  F[9] +  rotations[i][0][1] * F[10] + rotations[i][0][2] * F[11];
        f[d][1] -= rotations[i][1][0] *  F[9] +  rotations[i][1][1] * F[10] + rotations[i][1][2] * F[11];
        f[d][2] -= rotations[i][2][0] *  F[9] +  rotations[i][2][1] * F[10] + rotations[i][2][2] * F[11];
    }
}


template <class DataTypes>
void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::init()
{
    TetrahedronFEMForceField<DataTypes>::init();
    this->initMOR(this->_indexedElements->size(), notMuted());

    // The latent Vec1d state is only read during Gie collection (prepareECSW);
    // at runtime QuadraticManifoldMapping::applyJT does the contraction, so m_qState is unused.
    // initMOR has just populated m_nbRigid/m_nbDef from the bundle, so we know
    // the exact size to look for — see the size filter rationale in the finder.
    if (this->d_prepareECSW.getValue())
    {
        const std::size_t expected =
            static_cast<std::size_t>(this->m_nbRigid) + static_cast<std::size_t>(this->m_nbDef);
        m_qState = _find_vec1d_mstate_upwards(this->getContext(), expected);
        if (!m_qState)
        {
            msg_error(this) << "quadratic-manifold force field requires a Vec1d MechanicalObject "
                               "of size " << expected << " (the QuadraticManifoldMapping's input "
                               "latent state) in the parent chain. None found.";
        }
    }
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& p = d_x.getValue();

    SCOPED_TIMER("MORTetraQuadraticManifold::addForce");
    f.resize(p.size());

    if (this->needUpdateTopology)
    {
        this->reinit();
        this->needUpdateTopology = false;
    }

    // quadratic-manifold-specific: refresh the per-frame J(q_def) cache that updateGie consumes.
    // q is read from the cached Vec1d mstate pointer (parent chain).
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
    typename VecElement::const_iterator it, it0;
    switch(this->method)
    {
    case SMALL :
        for(it=this->_indexedElements->begin(), i = 0 ; it!=this->_indexedElements->end(); ++it,++i)
            accumulateForceSmall( f, p, it, i );
        break;
    case LARGE :
        if (d_performECSW.getValue()){
            it0=this->_indexedElements->begin();
            for( i = 0 ; i<m_RIDsize ;++i)
            {
                it = it0 + reducedIntegrationDomain(i);
                accumulateForceLarge( f, p, it, reducedIntegrationDomain(i) );
            }
        }
        else
        {
            for(it=this->_indexedElements->begin(), i = 0 ; it!=this->_indexedElements->end(); ++it,++i)
                accumulateForceLarge( f, p, it, i );
        }
        break;
    case POLAR :
        for(it=this->_indexedElements->begin(), i = 0 ; it!=this->_indexedElements->end(); ++it,++i)
            accumulateForcePolar( f, p, it, i );
        break;
    case SVD :
        for(it=this->_indexedElements->begin(), i = 0 ; it!=this->_indexedElements->end(); ++it,++i)
            accumulateForceSVD( f, p, it, i );
        break;
    }
    d_f.endEdit();

    this->updateVonMisesStress = true;
    this->saveGieFile(this->_indexedElements->size());
}


template<class DataTypes>
inline void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    auto dfAccessor = sofa::helper::getWriteAccessor(d_df);
    VecDeriv& df = dfAccessor.wref();

    const VecDeriv& dx = d_dx.getValue();
    df.resize(dx.size());

    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    unsigned int i;
    VecElement::const_iterator it, it0;
    if( this->method == SMALL )
    {
        for(it = this->_indexedElements->begin(), i = 0 ; it != this->_indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];
            Index c = (*it)[2];
            Index d = (*it)[3];
            applyStiffnessSmall( df,dx, i, a,b,c,d, kFactor );
        }
    }
    else
    {
        if (d_performECSW.getValue())
        {
            it0=this->_indexedElements->begin();
            for( i = 0 ; i<m_RIDsize ;++i)
            {
                it = it0 + reducedIntegrationDomain(i);
                Index a = (*it)[0];
                Index b = (*it)[1];
                Index c = (*it)[2];
                Index d = (*it)[3];
                applyStiffnessCorotational( df,dx, reducedIntegrationDomain(i), a,b,c,d, kFactor );
            }
        }
        else
        {
            for(it = this->_indexedElements->begin(), i = 0 ; it != this->_indexedElements->end() ; ++it, ++i)
            {
                Index a = (*it)[0];
                Index b = (*it)[1];
                Index c = (*it)[2];
                Index d = (*it)[3];
                applyStiffnessCorotational( df,dx, i, a,b,c,d, kFactor );
            }
        }
    }

    d_df.endEdit();
}


template<class DataTypes>
void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(this->d_componentState.getValue() == ComponentState::Invalid)
        return;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    if(this->needUpdateTopology)
    {
        this->reinit();
        this->needUpdateTopology = false;
    }

    vparams->drawTool()->saveLastState();
    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);
    vparams->drawTool()->disableLighting();

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    std::vector< type::Vec3 > points;
    std::vector< sofa::type::RGBAColor > colorVector;
    typename VecElement::const_iterator it, it0;

    it0=this->_indexedElements->begin();
    for(int i = 0 ; i<m_RIDsize ;++i)
    {
        it = it0 + reducedIntegrationDomain(i);
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];
        Index d = (*it)[3];
        Coord center = (x[a] + x[b] + x[c] + x[d]) * 0.125;
        Coord pa = x[a], pb = x[b], pc = x[c], pd = x[d];
        if ( ! vparams->displayFlags().getShowWireFrame() )
        {
            pa = (pa + center) * Real(0.6667);
            pb = (pb + center) * Real(0.6667);
            pc = (pc + center) * Real(0.6667);
            pd = (pd + center) * Real(0.6667);
        }
        sofa::type::RGBAColor color[4] = {
            sofa::type::RGBAColor(0.0, 0.0, 1.0, 1.0),
            sofa::type::RGBAColor(0.0, 0.5, 1.0, 1.0),
            sofa::type::RGBAColor(0.0, 1.0, 1.0, 1.0),
            sofa::type::RGBAColor(0.5, 1.0, 1.0, 1.0),
        };
        points.insert(points.end(), { pa, pb, pc });
        colorVector.insert(colorVector.end(), { color[0], color[0], color[0] });
        points.insert(points.end(), { pb, pc, pd });
        colorVector.insert(colorVector.end(), { color[1], color[1], color[1] });
        points.insert(points.end(), { pc, pd, pa });
        colorVector.insert(colorVector.end(), { color[2], color[2], color[2] });
        points.insert(points.end(), { pd, pa, pb });
        colorVector.insert(colorVector.end(), { color[3], color[3], color[3] });
    }
    vparams->drawTool()->drawTriangles(points, colorVector);

    vparams->drawTool()->restoreLastState();
}


template <class DataTypes>
void HyperReducedTetrahedronFEMForceFieldQuadraticManifold<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    StiffnessMatrix JKJt, RJKJtRt;
    sofa::type::Mat<3, 3, Real> localMatrix(type::NOINIT);

    static constexpr Transformation identity = []
    {
        Transformation i;
        i.identity();
        return i;
    }();

    constexpr auto S = DataTypes::deriv_total_size;
    constexpr auto N = Element::size();

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    sofa::Size tetraId = 0;

    typename VecElement::const_iterator it;
    auto it0=this->_indexedElements->begin();
    int nbElementsConsidered;
    if (!d_performECSW.getValue())
        nbElementsConsidered = this->_indexedElements->size();
    else
        nbElementsConsidered = m_RIDsize;

    for( unsigned int numElem = 0 ; numElem<nbElementsConsidered ;++numElem)
    {
        if (!d_performECSW.getValue())
            tetraId = numElem;
        else
            tetraId = reducedIntegrationDomain(numElem);
        it = it0 + tetraId;

        const auto& rotation = this->method == SMALL ? identity : rotations[tetraId];
        this->computeStiffnessMatrix(JKJt, RJKJtRt, materialsStiffnesses[tetraId], strainDisplacements[tetraId], rotation);

        for (sofa::Index n1 = 0; n1 < N; n1++)
        {
            for (sofa::Index n2 = 0; n2 < N; n2++)
            {
                RJKJtRt.getsub(S * n1, S * n2, localMatrix);
                if (!d_performECSW.getValue())
                    dfdx((*it)[n1] * S, (*it)[n2] * S) += -localMatrix;
                else
                    dfdx((*it)[n1] * S, (*it)[n2] * S) += -localMatrix*weights(tetraId);
            }
        }
    }
}


} // namespace sofa::component::forcefield
