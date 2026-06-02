/******************************************************************************
*  PreImageProjector_dump — standalone parity-test helper.
*
*  Loads an RBF kPCA bundle (+ preimage_config.json + mass_diagonal.txt) and a
*  latent state q, runs the pre-image head, and dumps:
*    psi.txt   (3N x 1)  = Ψ(q)              exact decode (RBF fixed point)
*    J.txt     (3N x m)  = J(q)              local-POD implicit-diff Jacobian
*    Kgeo.txt  (m  x m)  = G(f)              applyDJT geometric stiffness via
*                          directional FD of J:  G[:,l] = [(J(q+ε e_l)−J(q))/ε]ᵀ f
*  The src/kpca/preimage.py reference is diffed against these at ~1e-6.
*
*  Usage:
*    PreImageProjector_dump <bundle_dir> <q_path> <u_prev_path> <f_path> <out_dir> <eps>
******************************************************************************/
#include <ModelOrderReduction/component/kernel/PreImageProjector.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

static void write_matrix(const std::string& path, const Eigen::MatrixXd& M)
{
    std::ofstream f(path);
    f << M.rows() << " " << M.cols() << "\n";
    f << std::scientific;
    f.precision(17);
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            f << M(i, j) << (j + 1 < M.cols() ? ' ' : '\n');
}

static Eigen::MatrixXd read_matrix(const std::string& path)
{
    sofa::component::loader::MatrixLoader<Eigen::MatrixXd> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXd M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to read: " + path);
    return M;
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::cerr << "usage: " << argv[0]
                  << " <bundle_dir> <q_path> <u_prev_path> <f_path> <out_dir> <eps>\n";
        return 2;
    }
    const std::string bundle_dir = argv[1];
    const std::string q_path     = argv[2];
    const std::string uprev_path = argv[3];
    const std::string f_path     = argv[4];
    const std::string out_dir    = argv[5];
    const double eps             = std::stod(argv[6]);

    std::filesystem::create_directories(out_dir);

    auto proj = sofa::component::kernel::loadPreImageProjectorFromBundle(bundle_dir);

    const Eigen::VectorXd q     = read_matrix(q_path).col(0);
    const Eigen::VectorXd uprev = read_matrix(uprev_path).col(0);
    const Eigen::VectorXd f     = read_matrix(f_path).col(0);

    if (q.size() != static_cast<Eigen::Index>(proj->nbModes()))
    {
        std::cerr << "q has " << q.size() << " entries; expected m="
                  << proj->nbModes() << "\n";
        return 2;
    }

    const Eigen::VectorXd u_init = proj->uInit(q);
    const Eigen::VectorXd psi    = proj->solve(q, u_init, uprev);
    const Eigen::MatrixXd J      = proj->jacobianLocal(q, u_init, uprev);

    const Eigen::Index m = static_cast<Eigen::Index>(proj->nbModes());
    Eigen::MatrixXd Kgeo(m, m);
    for (Eigen::Index l = 0; l < m; ++l)
    {
        Eigen::VectorXd ql = q;
        ql(l) += eps;
        const Eigen::VectorXd u_init_l = proj->uInit(ql);
        const Eigen::MatrixXd Jl = proj->jacobianLocal(ql, u_init_l, uprev);
        Kgeo.col(l) = ((Jl - J) / eps).transpose() * f;       // (m,)
    }

    write_matrix(out_dir + "/psi.txt",  psi);
    write_matrix(out_dir + "/J.txt",    J);
    write_matrix(out_dir + "/Kgeo.txt", Kgeo);

    std::cout << "dumped pre-image head: kernel=" << proj->kernelName()
              << "  3N=" << proj->nbDofs()
              << "  T="  << proj->nbSnapshots()
              << "  m="  << proj->nbModes()
              << "  eta=" << proj->eta() << "  eta_t=" << proj->eta_t()
              << "  r=" << proj->r() << "  sigma=" << proj->sigma() << "\n";
    return 0;
}
