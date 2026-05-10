#include "spectral_bandit/patchification.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace spectral_bandit {
namespace {

int initial_center(const Mesh& mesh, std::mt19937* rng) {
  if (rng != nullptr) {
    std::uniform_int_distribution<int> dis(0, mesh.n_vertices() - 1);
    return dis(*rng);
  }

  Eigen::Vector3d c = mesh.vertices.colwise().mean();
  int best = 0;
  double best_d = -1.0;
  for (int i = 0; i < mesh.n_vertices(); ++i) {
    const double d = (mesh.vertices.row(i).transpose() - c).norm();
    if (d > best_d) {
      best_d = d;
      best = i;
    }
  }
  return best;
}

int majority_label(int a, int b, int c) {
  if (a == b || a == c) {
    return a;
  }
  if (b == c) {
    return b;
  }
  return a;
}

Eigen::VectorXi compute_patch_center_vertices(const Mesh& mesh, const Eigen::VectorXd& mass_diag,
                                              const std::vector<Eigen::VectorXi>& patches) {
  Eigen::VectorXi centers(static_cast<Eigen::Index>(patches.size()));

  for (Eigen::Index j = 0; j < centers.size(); ++j) {
    const auto& idx = patches[static_cast<std::size_t>(j)];
    if (idx.size() == 0) {
      centers(j) = 0;
      continue;
    }

    double wsum = 0.0;
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (Eigen::Index t = 0; t < idx.size(); ++t) {
      const int vi = idx(t);
      const double w = mass_diag(vi);
      c += w * mesh.vertices.row(vi).transpose();
      wsum += w;
    }

    if (wsum <= 0.0) {
      c.setZero();
      for (Eigen::Index t = 0; t < idx.size(); ++t) {
        c += mesh.vertices.row(idx(t)).transpose();
      }
      c /= static_cast<double>(idx.size());
    } else {
      c /= wsum;
    }

    int best_v = idx(0);
    double best_d = std::numeric_limits<double>::infinity();
    for (Eigen::Index t = 0; t < idx.size(); ++t) {
      const int vi = idx(t);
      const double d = (mesh.vertices.row(vi).transpose() - c).norm();
      if (d < best_d) {
        best_d = d;
        best_v = vi;
      }
    }
    centers(j) = best_v;
  }

  return centers;
}

}  // namespace

PatchificationResult geodesic_patchify(const Mesh& mesh, const HeatMethodGeodesics& geodesics,
                                       const Eigen::VectorXd& mass_diag, int n_patches, std::optional<int> seed) {
  if (n_patches <= 0) {
    throw std::runtime_error("geodesic_patchify: n_patches must be >= 1");
  }
  if (n_patches > mesh.n_vertices()) {
    throw std::runtime_error("geodesic_patchify: n_patches cannot exceed number of vertices");
  }

  std::optional<std::mt19937> rng_opt;
  if (seed.has_value()) {
    rng_opt.emplace(static_cast<std::uint32_t>(*seed));
  }

  std::vector<int> centers;
  centers.reserve(static_cast<std::size_t>(n_patches));
  centers.push_back(initial_center(mesh, rng_opt ? &(*rng_opt) : nullptr));

  Eigen::VectorXd d_min = geodesics.distance_from(centers[0]);

  while (static_cast<int>(centers.size()) < n_patches) {
    Eigen::Index best_idx = 0;
    d_min.maxCoeff(&best_idx);
    int c_new = static_cast<int>(best_idx);

    if (std::find(centers.begin(), centers.end(), c_new) != centers.end()) {
      std::vector<int> order(static_cast<std::size_t>(d_min.size()));
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&d_min](int a, int b) { return d_min(a) > d_min(b); });
      for (int cand : order) {
        if (std::find(centers.begin(), centers.end(), cand) == centers.end()) {
          c_new = cand;
          break;
        }
      }
    }

    centers.push_back(c_new);
    Eigen::VectorXd d_new = geodesics.distance_from(c_new);
    d_min = d_min.cwiseMin(d_new);
  }

  Eigen::VectorXi centers_arr(n_patches);
  for (int i = 0; i < n_patches; ++i) {
    centers_arr(i) = centers[static_cast<std::size_t>(i)];
  }

  std::vector<int> center_vec(static_cast<std::size_t>(centers_arr.size()));
  for (Eigen::Index i = 0; i < centers_arr.size(); ++i) {
    center_vec[static_cast<std::size_t>(i)] = centers_arr(i);
  }
  Eigen::MatrixXd d_all = geodesics.distance_matrix(center_vec);

  Eigen::VectorXi vertex_labels(mesh.n_vertices());
  for (int i = 0; i < mesh.n_vertices(); ++i) {
    Eigen::Index argmin = 0;
    d_all.row(i).minCoeff(&argmin);
    vertex_labels(i) = static_cast<int>(argmin);
  }

  std::vector<std::vector<int>> patch_lists(static_cast<std::size_t>(n_patches));
  for (int i = 0; i < vertex_labels.size(); ++i) {
    patch_lists[static_cast<std::size_t>(vertex_labels(i))].push_back(i);
  }

  std::vector<Eigen::VectorXi> patches;
  patches.reserve(static_cast<std::size_t>(n_patches));
  for (int j = 0; j < n_patches; ++j) {
    const auto& p = patch_lists[static_cast<std::size_t>(j)];
    Eigen::VectorXi idx(static_cast<Eigen::Index>(p.size()));
    for (Eigen::Index t = 0; t < idx.size(); ++t) {
      idx(t) = p[static_cast<std::size_t>(t)];
    }
    patches.push_back(idx);
  }

  Eigen::VectorXi face_labels(mesh.n_faces());
  for (int fi = 0; fi < mesh.n_faces(); ++fi) {
    const int a = vertex_labels(mesh.faces(fi, 0));
    const int b = vertex_labels(mesh.faces(fi, 1));
    const int c = vertex_labels(mesh.faces(fi, 2));
    face_labels(fi) = majority_label(a, b, c);
  }

  Eigen::VectorXi center_vertices = compute_patch_center_vertices(mesh, mass_diag, patches);

  return PatchificationResult{
      .centers = centers_arr,
      .vertex_labels = vertex_labels,
      .face_labels = face_labels,
      .patches = patches,
      .center_vertices = center_vertices,
  };
}

}  // namespace spectral_bandit
