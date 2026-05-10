#include "spectral_bandit/spectral.hpp"

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spectral_bandit {
namespace {

Eigen::VectorXd sparse_diag(const SparseMatrix& m) {
  Eigen::VectorXd d(m.rows());
  for (int i = 0; i < m.rows(); ++i) {
    d(i) = m.coeff(i, i);
  }
  return d;
}

SparseMatrix build_normalized_laplacian(const SparseMatrix& L, const Eigen::VectorXd& mass_diag) {
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(L.nonZeros()));

  Eigen::VectorXd inv_sqrt = mass_diag.array().sqrt().inverse();

  for (int k = 0; k < L.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(L, k); it; ++it) {
      const int i = it.row();
      const int j = it.col();
      const double v = it.value() * inv_sqrt(i) * inv_sqrt(j);
      if (std::abs(v) > 0.0) {
        trips.emplace_back(i, j, v);
      }
    }
  }

  SparseMatrix A(L.rows(), L.cols());
  A.setFromTriplets(trips.begin(), trips.end());
  A = (A + SparseMatrix(A.transpose())) * 0.5;
  A.makeCompressed();
  return A;
}

}  // namespace

SpectralEmbeddingResult spectral_embedding(const SparseMatrix& laplacian, const SparseMatrix& mass_matrix, int k,
                                           bool drop_first) {
  if (k <= 0) {
    throw std::runtime_error("spectral_embedding: k must be >= 1");
  }
  const int n = laplacian.rows();
  if (laplacian.cols() != n || mass_matrix.rows() != n || mass_matrix.cols() != n) {
    throw std::runtime_error("spectral_embedding: dimension mismatch");
  }
  if (k >= n) {
    throw std::runtime_error("spectral_embedding: k must be < n_vertices");
  }

  const int target = drop_first ? (k + 1) : k;

  int ncv = std::min(n, std::max(target + 8, std::max(40, 8 * target)));
  ncv = std::max(ncv, target + 1);
  constexpr int maxit = 8000;
  constexpr double tol = 1e-8;
  constexpr double shift_sigma = 1e-6;

  Eigen::VectorXd evals;
  Eigen::MatrixXd evecs;
  bool ok = false;
  bool solved_generalized = false;

  // 1) Robust generalized solver directly on L u = lambda M u.
  try {
    Spectra::SparseSymMatProd<double> opA(laplacian);
    Spectra::SparseCholesky<double> opB(mass_matrix);
    Spectra::SymGEigsSolver<decltype(opA), decltype(opB), Spectra::GEigsMode::Cholesky> geigs(opA, opB, target, ncv);
    geigs.init();
    const int nconv = geigs.compute(Spectra::SortRule::SmallestAlge, maxit, tol, Spectra::SortRule::SmallestAlge);
    if (geigs.info() == Spectra::CompInfo::Successful && nconv >= target) {
      evals = geigs.eigenvalues();
      evecs = geigs.eigenvectors();
      ok = true;
      solved_generalized = true;
    }
  } catch (...) {
    ok = false;
  }

  // 2) Fallback on normalized eigenproblem if generalized mode fails.
  const Eigen::VectorXd mdiag = sparse_diag(mass_matrix);
  const SparseMatrix A = build_normalized_laplacian(laplacian, mdiag);

  try {
    if (!ok) {
      Spectra::SparseSymShiftSolve<double> op(A);
      Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op, target, ncv, shift_sigma);
      eigs.init();
      const int nconv = eigs.compute(Spectra::SortRule::LargestMagn, maxit, tol, Spectra::SortRule::LargestAlge);
      if (eigs.info() == Spectra::CompInfo::Successful && nconv >= target) {
        evals = eigs.eigenvalues();
        evecs = eigs.eigenvectors();
        ok = true;
        solved_generalized = false;
      }
    }
  } catch (...) {
    if (!ok) {
      ok = false;
    }
  }

  if (!ok) {
    Spectra::SparseSymMatProd<double> op(A);
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, target, ncv);
    eigs.init();
    const int nconv = eigs.compute(Spectra::SortRule::SmallestMagn, maxit, tol, Spectra::SortRule::SmallestAlge);
    if (eigs.info() != Spectra::CompInfo::Successful || nconv < target) {
      throw std::runtime_error("spectral_embedding: eigensolver failed to converge");
    }
    evals = eigs.eigenvalues();
    evecs = eigs.eigenvectors();
    solved_generalized = false;
  }

  std::vector<int> order(static_cast<std::size_t>(evals.size()));
  for (int i = 0; i < evals.size(); ++i) {
    order[static_cast<std::size_t>(i)] = i;
  }
  std::sort(order.begin(), order.end(), [&evals](int a, int b) { return evals(a) < evals(b); });

  Eigen::VectorXd eval_sorted(evals.size());
  Eigen::MatrixXd y_sorted(evecs.rows(), evecs.cols());
  for (Eigen::Index j = 0; j < evals.size(); ++j) {
    const int src = order[static_cast<std::size_t>(j)];
    eval_sorted(j) = evals(src);
    y_sorted.col(j) = evecs.col(src);
  }

  Eigen::MatrixXd u = y_sorted;
  if (!solved_generalized) {
    const Eigen::VectorXd inv_sqrt = mdiag.array().sqrt().inverse();
    for (Eigen::Index i = 0; i < u.rows(); ++i) {
      u.row(i) *= inv_sqrt(i);
    }
  }

  if (!u.allFinite()) {
    throw std::runtime_error("spectral_embedding: invalid eigenvectors");
  }

  int start_col = drop_first ? 1 : 0;
  if (start_col >= u.cols()) {
    throw std::runtime_error("spectral_embedding: not enough eigenvectors after drop_first");
  }

  int take = std::min(k, static_cast<int>(u.cols()) - start_col);
  Eigen::MatrixXd emb = u.block(0, start_col, u.rows(), take);
  Eigen::VectorXd lam = eval_sorted.segment(start_col, take);

  return SpectralEmbeddingResult{
      .vertex_embeddings = emb,
      .eigenvalues = lam,
      .eigenvectors = emb,
  };
}

Eigen::MatrixXd patch_embeddings(const Eigen::MatrixXd& vertex_embeddings, const Eigen::VectorXi& vertex_labels,
                                 const Eigen::VectorXd& mass_diag, int n_patches) {
  if (n_patches <= 0) {
    throw std::runtime_error("patch_embeddings: n_patches must be >= 1");
  }
  if (vertex_embeddings.rows() != vertex_labels.size() || vertex_labels.size() != mass_diag.size()) {
    throw std::runtime_error("patch_embeddings: size mismatch");
  }

  const int d = static_cast<int>(vertex_embeddings.cols());
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_patches, d);

  for (int j = 0; j < n_patches; ++j) {
    double wsum = 0.0;
    Eigen::RowVectorXd acc = Eigen::RowVectorXd::Zero(d);
    int cnt = 0;

    for (Eigen::Index i = 0; i < vertex_labels.size(); ++i) {
      if (vertex_labels(i) != j) {
        continue;
      }
      const double w = mass_diag(i);
      acc += w * vertex_embeddings.row(i);
      wsum += w;
      cnt += 1;
    }

    if (cnt == 0) {
      continue;
    }

    if (wsum <= 0.0) {
      Eigen::RowVectorXd mean = Eigen::RowVectorXd::Zero(d);
      for (Eigen::Index i = 0; i < vertex_labels.size(); ++i) {
        if (vertex_labels(i) == j) {
          mean += vertex_embeddings.row(i);
        }
      }
      out.row(j) = mean / static_cast<double>(cnt);
    } else {
      out.row(j) = acc / wsum;
    }
  }

  return out;
}

}  // namespace spectral_bandit
