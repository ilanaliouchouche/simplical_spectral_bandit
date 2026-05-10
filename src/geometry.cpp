#include "spectral_bandit/geometry.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace spectral_bandit {

std::pair<Eigen::VectorXd, Eigen::MatrixXd> face_areas_and_normals(const Mesh& mesh) {
  const int nf = mesh.n_faces();
  Eigen::VectorXd areas(nf);
  Eigen::MatrixXd normals(nf, 3);

  for (int fi = 0; fi < nf; ++fi) {
    const int i = mesh.faces(fi, 0);
    const int j = mesh.faces(fi, 1);
    const int k = mesh.faces(fi, 2);

    const Eigen::Vector3d vi = mesh.vertices.row(i);
    const Eigen::Vector3d vj = mesh.vertices.row(j);
    const Eigen::Vector3d vk = mesh.vertices.row(k);

    const Eigen::Vector3d n = (vj - vi).cross(vk - vi);
    const double n_norm = n.norm();
    areas(fi) = 0.5 * n_norm;

    if (n_norm > kEps) {
      normals.row(fi) = (n / n_norm).transpose();
    } else {
      normals.row(fi).setZero();
    }
  }

  return {areas, normals};
}

std::pair<Eigen::VectorXd, SparseMatrix> mass_matrix(const Mesh& mesh) {
  const auto [areas, _] = face_areas_and_normals(mesh);
  (void)_;

  Eigen::VectorXd m = Eigen::VectorXd::Zero(mesh.n_vertices());
  for (int fi = 0; fi < mesh.n_faces(); ++fi) {
    const double a3 = areas(fi) / 3.0;
    m(mesh.faces(fi, 0)) += a3;
    m(mesh.faces(fi, 1)) += a3;
    m(mesh.faces(fi, 2)) += a3;
  }

  for (int i = 0; i < m.size(); ++i) {
    m(i) = std::max(m(i), kEps);
  }

  SparseMatrix M(m.size(), m.size());
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(m.size()));
  for (int i = 0; i < m.size(); ++i) {
    trips.emplace_back(i, i, m(i));
  }
  M.setFromTriplets(trips.begin(), trips.end());

  return {m, M};
}

namespace {

double cotangent(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const double cross_norm = a.cross(b).norm();
  const double dot = a.dot(b);
  return dot / std::max(cross_norm, kEps);
}

void add_symmetric_edge(std::vector<Eigen::Triplet<double>>& trips, Eigen::VectorXd& diag, int a, int b, double w) {
  trips.emplace_back(a, b, -w);
  trips.emplace_back(b, a, -w);
  diag(a) += w;
  diag(b) += w;
}

}  // namespace

SparseMatrix cotangent_laplacian(const Mesh& mesh) {
  const int n = mesh.n_vertices();
  Eigen::VectorXd diag = Eigen::VectorXd::Zero(n);
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(mesh.n_faces() * 6 + n));

  for (int fi = 0; fi < mesh.n_faces(); ++fi) {
    const int i = mesh.faces(fi, 0);
    const int j = mesh.faces(fi, 1);
    const int k = mesh.faces(fi, 2);

    const Eigen::Vector3d vi = mesh.vertices.row(i);
    const Eigen::Vector3d vj = mesh.vertices.row(j);
    const Eigen::Vector3d vk = mesh.vertices.row(k);

    const double cot_i = cotangent(vj - vi, vk - vi);
    const double cot_j = cotangent(vk - vj, vi - vj);
    const double cot_k = cotangent(vi - vk, vj - vk);

    const double w_ij = 0.5 * cot_k;
    const double w_jk = 0.5 * cot_i;
    const double w_ki = 0.5 * cot_j;

    add_symmetric_edge(trips, diag, i, j, w_ij);
    add_symmetric_edge(trips, diag, j, k, w_jk);
    add_symmetric_edge(trips, diag, k, i, w_ki);
  }

  for (int i = 0; i < n; ++i) {
    trips.emplace_back(i, i, diag(i));
  }

  SparseMatrix L(n, n);
  L.setFromTriplets(trips.begin(), trips.end());

  // Enforce symmetry to reduce numerical drift.
  SparseMatrix Lt = SparseMatrix(L.transpose());
  L = (L + Lt) * 0.5;
  L.makeCompressed();
  return L;
}

AdjacencyList build_graph_adjacency(const Mesh& mesh) {
  const int n = mesh.n_vertices();

  // Keep shortest duplicate edge length if mesh has repeated edges.
  std::vector<std::unordered_map<int, double>> tmp(static_cast<std::size_t>(n));

  for (int fi = 0; fi < mesh.n_faces(); ++fi) {
    const std::array<std::pair<int, int>, 3> edges = {
        std::make_pair(mesh.faces(fi, 0), mesh.faces(fi, 1)),
        std::make_pair(mesh.faces(fi, 1), mesh.faces(fi, 2)),
        std::make_pair(mesh.faces(fi, 2), mesh.faces(fi, 0)),
    };

    for (const auto& e : edges) {
      const int a = e.first;
      const int b = e.second;
      const Eigen::Vector3d va = mesh.vertices.row(a);
      const Eigen::Vector3d vb = mesh.vertices.row(b);
      const double w = (va - vb).norm();

      auto update_min = [w](std::unordered_map<int, double>& m, int key) {
        auto it = m.find(key);
        if (it == m.end()) {
          m.emplace(key, w);
          return;
        }
        if (w < it->second) {
          it->second = w;
        }
      };

      update_min(tmp[static_cast<std::size_t>(a)], b);
      update_min(tmp[static_cast<std::size_t>(b)], a);
    }
  }

  AdjacencyList adj(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    adj[static_cast<std::size_t>(i)].reserve(tmp[static_cast<std::size_t>(i)].size());
    for (const auto& [j, w] : tmp[static_cast<std::size_t>(i)]) {
      adj[static_cast<std::size_t>(i)].emplace_back(j, w);
    }
  }
  return adj;
}

Eigen::MatrixXd vertex_normals(const Mesh& mesh) {
  const auto [areas, f_normals] = face_areas_and_normals(mesh);

  Eigen::MatrixXd vn = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);

  for (int fi = 0; fi < mesh.n_faces(); ++fi) {
    const Eigen::Vector3d weighted = f_normals.row(fi).transpose() * areas(fi);
    vn.row(mesh.faces(fi, 0)) += weighted.transpose();
    vn.row(mesh.faces(fi, 1)) += weighted.transpose();
    vn.row(mesh.faces(fi, 2)) += weighted.transpose();
  }

  for (int i = 0; i < vn.rows(); ++i) {
    const double n = vn.row(i).norm();
    if (n > kEps) {
      vn.row(i) /= n;
    }
  }

  return vn;
}

double average_edge_length(const Mesh& mesh) {
  struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32U) ^ static_cast<std::size_t>(p.second);
    }
  };

  std::unordered_map<std::pair<int, int>, double, PairHash> unique;

  auto add_edge = [&](int a, int b) {
    if (a > b) {
      std::swap(a, b);
    }
    const auto key = std::make_pair(a, b);
    if (unique.find(key) == unique.end()) {
      const Eigen::Vector3d va = mesh.vertices.row(a);
      const Eigen::Vector3d vb = mesh.vertices.row(b);
      unique.emplace(key, (va - vb).norm());
    }
  };

  for (int fi = 0; fi < mesh.n_faces(); ++fi) {
    add_edge(mesh.faces(fi, 0), mesh.faces(fi, 1));
    add_edge(mesh.faces(fi, 1), mesh.faces(fi, 2));
    add_edge(mesh.faces(fi, 2), mesh.faces(fi, 0));
  }

  if (unique.empty()) {
    return 0.0;
  }

  double s = 0.0;
  for (const auto& [_, w] : unique) {
    (void)_;
    s += w;
  }
  return s / static_cast<double>(unique.size());
}

}  // namespace spectral_bandit
