/******************************************************************************
*  HyperReducedHelperKPCA_dump — Stage 5a parity gate helper.
*
*  Loads a kPCA bundle into HyperReducedHelperKPCA, runs prepareFrame(u) +
*  projectOneElement<Vec3d>(indexList, contrib) for a single fake element,
*  writes the resulting (m,) GieUnit vector to disk. A Python test exercises
*  the NumPy reference (KernelProjector.project_element) with the same inputs
*  and diffs at machine precision.
*
*  Usage:
*    HyperReducedHelperKPCA_dump <bundle_dir> <u_path> <indices_path> <contrib_path> <out_path>
*
*  All matrix files follow the plugin's MatrixLoader convention:
*      first line: "nbRows nbCols"
*      then nbRows rows of whitespace-separated floats (or ints for indices)
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperKPCA.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/defaulttype/VecTypes.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using modelorderreduction::HyperReducedHelperKPCA;
using sofa::component::loader::MatrixLoader;

static Eigen::MatrixXd read_matrix_d(const std::string& path)
{
    MatrixLoader<Eigen::MatrixXd> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXd M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to read: " + path);
    return M;
}

static Eigen::MatrixXi read_matrix_i(const std::string& path)
{
    MatrixLoader<Eigen::MatrixXi> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXi M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to read: " + path);
    return M;
}

static void write_vector(const std::string& path, const Eigen::VectorXd& v)
{
    std::ofstream f(path);
    f << v.size() << " 1\n";
    f << std::scientific;
    f.precision(17);
    for (Eigen::Index i = 0; i < v.size(); ++i)
        f << v(i) << "\n";
}


int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: " << argv[0]
                  << " <bundle_dir> <u_path> <indices_path> <contrib_path> <out_path>\n";
        return 2;
    }

    HyperReducedHelperKPCA helper;
    helper.d_prepareECSW.setValue(true);
    helper.d_kernelBundle.setValue(argv[1]);
    helper.d_nbTrainingSet.setValue(1);  // any nonzero — we only exercise projectOneElement
    helper.d_periodSaveGIE.setValue(1);

    // nbElements = 1 (a single fake element).
    helper.initMOR(1u, /*printLog*/ false);

    // Load inputs.
    Eigen::MatrixXd u_mat      = read_matrix_d(argv[2]);     // (3N, 1)
    Eigen::MatrixXi idx_mat    = read_matrix_i(argv[3]);     // (V, 1)
    Eigen::MatrixXd contr_mat  = read_matrix_d(argv[4]);     // (V, 3)
    if (u_mat.cols() != 1)       throw std::runtime_error("u must be (3N, 1)");
    if (idx_mat.cols() != 1)     throw std::runtime_error("indices must be (V, 1)");
    if (contr_mat.cols() != 3)   throw std::runtime_error("contrib must be (V, 3)");
    if (idx_mat.rows() != contr_mat.rows())
        throw std::runtime_error("indices rows != contrib rows");
    const Eigen::VectorXd u = u_mat.col(0);
    const unsigned V = static_cast<unsigned>(idx_mat.rows());

    std::vector<unsigned int> indexList(V);
    for (unsigned i = 0; i < V; ++i)
        indexList[i] = static_cast<unsigned>(idx_mat(i, 0));

    using Vec3d = sofa::defaulttype::Vec3dTypes;
    std::vector<Vec3d::Deriv> contrib(V);
    for (unsigned i = 0; i < V; ++i)
        contrib[i] = Vec3d::Deriv(contr_mat(i, 0), contr_mat(i, 1), contr_mat(i, 2));

    // Do the projection.
    helper.prepareFrame(u);
    Eigen::VectorXd GieUnit = helper.projectOneElement<Vec3d>(indexList, contrib);
    write_vector(argv[5], GieUnit);

    std::cout << "dumped GieUnit (m=" << GieUnit.size()
              << ") for bundle=" << argv[1]
              << " kernel=" << helper.m_projector->kernelName() << "\n";
    return 0;
}
