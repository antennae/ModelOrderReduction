/******************************************************************************
*  KernelProjector_dump — standalone parity-test helper.
*
*  Reads a kPCA bundle + a "u" test vector from disk, runs the three kernel
*  primitives (kernel_matrix, grad_u, apply_Ginv) on the bundle's snapshots
*  with that u, and dumps the outputs as MatrixLoader-compatible text files.
*  A Python test exercises the NumPy reference with the same inputs and diffs
*  the outputs at 1e-12.
*
*  Usage:
*    KernelProjector_dump <bundle_dir> <u_path> <out_dir>
******************************************************************************/
#include <ModelOrderReduction/component/kernel/KernelProjector.h>
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
    {
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            f << M(i, j) << (j + 1 < M.cols() ? ' ' : '\n');
    }
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
    if (argc != 4)
    {
        std::cerr << "usage: " << argv[0] << " <bundle_dir> <u_path> <out_dir>\n";
        return 2;
    }
    const std::string bundle_dir = argv[1];
    const std::string u_path     = argv[2];
    const std::string out_dir    = argv[3];

    std::filesystem::create_directories(out_dir);

    auto proj = sofa::component::kernel::loadKernelProjectorFromBundle(bundle_dir);

    Eigen::MatrixXd u_mat = read_matrix(u_path);
    if (u_mat.cols() != 1 || u_mat.rows() != proj->nbDofs())
    {
        std::cerr << "u must have shape (" << proj->nbDofs() << ", 1); got ("
                  << u_mat.rows() << ", " << u_mat.cols() << ")\n";
        return 2;
    }
    Eigen::VectorXd u = u_mat.col(0);

    const auto& snapshots = proj->snapshots();

    Eigen::MatrixXd K       = proj->kernel_matrix(snapshots, snapshots);
    Eigen::MatrixXd gradu   = proj->grad_u(u, snapshots);
    Eigen::MatrixXd Ginv_sn = proj->apply_Ginv(u, snapshots);

    write_matrix(out_dir + "/K.txt",       K);
    write_matrix(out_dir + "/grad_u.txt",  gradu);
    write_matrix(out_dir + "/Ginv_sn.txt", Ginv_sn);

    std::cout << "dumped for kernel=" << proj->kernelName()
              << "  T=" << proj->nbSnapshots()
              << "  3N=" << proj->nbDofs()
              << "  m=" << proj->nbModes() << "\n";
    return 0;
}
