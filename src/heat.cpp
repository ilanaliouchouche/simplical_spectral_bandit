#include "spectral_bandit/heat.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <stdexcept>

namespace spectral_bandit {

namespace {

double median(Eigen::VectorXd v) {
  std::vector<double> data(static_cast<std::size_t>(v.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    data[static_cast<std::size_t>(i)] = v(i);
  }
  const std::size_t n = data.size();
  if (n == 0) {
    return 0.0;
  }

  const std::size_t mid = n / 2;
  std::nth_element(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(mid), data.end());
  double m = data[mid];
  if (n % 2 == 0) {
    std::nth_element(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(mid - 1), data.end());
    m = 0.5 * (m + data[mid - 1]);
  }
  return m;
}

SparseMatrix diag_sparse(const Eigen::VectorXd& d) {
  SparseMatrix D(d.size(), d.size());
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(d.size()));
  for (Eigen::Index i = 0; i < d.size(); ++i) {
    trips.emplace_back(i, i, d(i));
  }
  D.setFromTriplets(trips.begin(), trips.end());
  return D;
}

}  // namespace

HeatMethodGeodesics::HeatMethodGeodesics(const Mesh& mesh, const SparseMatrix& laplacian,
                                         const Eigen::VectorXd& mass_diag, double t, double regularization)
    : mesh_(mesh), laplacian_(laplacian), mass_diag_(mass_diag), t_(t) {
  if (mesh_.n_vertices() <= 0 || mesh_.n_faces() <= 0) {
    throw std::runtime_error("HeatMethodGeodesics: empty mesh");
  }

  if (mass_diag_.size() != mesh_.n_vertices()) {
    throw std::runtime_error("HeatMethodGeodesics: mass_diag size mismatch");
  }

  if (t_ <= 0.0) {
    const double h = average_edge_length(mesh_);
    t_ = std::max(h * h, 1e-10);
  }

  const SparseMatrix M = diag_sparse(mass_diag_);
  heat_matrix_ = (M + t_ * laplacian_).eval();
  poisson_matrix_ = (laplacian_ + regularization * M).eval();

  heat_solver_.compute(heat_matrix_);
  if (heat_solver_.info() != Eigen::Success) {
    throw std::runtime_error("HeatMethodGeodesics: heat solver factorization failed");
  }

  poisson_solver_.compute(poisson_matrix_);
  if (poisson_solver_.info() != Eigen::Success) {
    throw std::runtime_error("HeatMethodGeodesics: poisson solver factorization failed");
  }

  grad_cache_ = build_gradient_cache();
}

HeatMethodGeodesics::FaceGradientCache HeatMethodGeodesics::build_gradient_cache() const {
  const int nf = mesh_.n_faces();
  FaceGradientCache cache;
  cache.g0.resize(nf, 3);
  cache.g1.resize(nf, 3);
  cache.g2.resize(nf, 3);

  const auto [areas, face_normals] = face_areas_and_normals(mesh_);
  cache.areas = areas.cwiseMax(kEps);

  for (int fi = 0; fi < nf; ++fi) {
    const int i = mesh_.faces(fi, 0);
    const int j = mesh_.faces(fi, 1);
    const int k = mesh_.faces(fi, 2);

    const Eigen::Vector3d vi = mesh_.vertices.row(i);
    const Eigen::Vector3d vj = mesh_.vertices.row(j);
    const Eigen::Vector3d vk = mesh_.vertices.row(k);

    const Eigen::Vector3d n_hat = face_normals.row(fi);
    const double area2 = 2.0 * std::max(areas(fi), kEps);

    cache.g0.row(fi) = (n_hat.cross(vk - vj) / area2).transpose();
    cache.g1.row(fi) = (n_hat.cross(vi - vk) / area2).transpose();
    cache.g2.row(fi) = (n_hat.cross(vj - vi) / area2).transpose();
  }

  return cache;
}

Eigen::VectorXd HeatMethodGeodesics::heat_step(int source) const {
  if (source < 0 || source >= mesh_.n_vertices()) {
    throw std::out_of_range("HeatMethodGeodesics::heat_step: source out of range");
  }

  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(mesh_.n_vertices());
  u0(source) = 1.0;

  Eigen::VectorXd rhs = mass_diag_.array() * u0.array();
  Eigen::VectorXd u = heat_solver_.solve(rhs);
  if (heat_solver_.info() != Eigen::Success) {
    throw std::runtime_error("HeatMethodGeodesics::heat_step: solve failed");
  }
  return u;
}

Eigen::MatrixXd HeatMethodGeodesics::compute_normalized_vector_field(const Eigen::VectorXd& u) const {
  Eigen::MatrixXd x(mesh_.n_faces(), 3);

  for (int fi = 0; fi < mesh_.n_faces(); ++fi) {
    const int i = mesh_.faces(fi, 0);
    const int j = mesh_.faces(fi, 1);
    const int k = mesh_.faces(fi, 2);

    Eigen::Vector3d grad_u = u(i) * grad_cache_.g0.row(fi).transpose() + u(j) * grad_cache_.g1.row(fi).transpose() +
                             u(k) * grad_cache_.g2.row(fi).transpose();

    const double n = std::max(grad_u.norm(), kEps);
    x.row(fi) = (-grad_u / n).transpose();
  }

  return x;
}

Eigen::VectorXd HeatMethodGeodesics::divergence(const Eigen::MatrixXd& x) const {
  Eigen::VectorXd div = Eigen::VectorXd::Zero(mesh_.n_vertices());

  for (int fi = 0; fi < mesh_.n_faces(); ++fi) {
    const Eigen::Vector3d xf = x.row(fi);
    const double a = grad_cache_.areas(fi);

    const double c0 = -a * grad_cache_.g0.row(fi).dot(xf);
    const double c1 = -a * grad_cache_.g1.row(fi).dot(xf);
    const double c2 = -a * grad_cache_.g2.row(fi).dot(xf);

    div(mesh_.faces(fi, 0)) += c0;
    div(mesh_.faces(fi, 1)) += c1;
    div(mesh_.faces(fi, 2)) += c2;
  }

  return div;
}

Eigen::VectorXd HeatMethodGeodesics::distance_from(int source) const {
  if (source < 0 || source >= mesh_.n_vertices()) {
    throw std::out_of_range("HeatMethodGeodesics::distance_from: source out of range");
  }

  const Eigen::VectorXd u = heat_step(source);
  const Eigen::MatrixXd x = compute_normalized_vector_field(u);
  const Eigen::VectorXd div = divergence(x);

  Eigen::VectorXd phi = poisson_solver_.solve(div);
  if (poisson_solver_.info() != Eigen::Success) {
    throw std::runtime_error("HeatMethodGeodesics::distance_from: poisson solve failed");
  }

  phi.array() -= phi(source);
  if (median(phi) < 0.0) {
    phi = -phi;
  }
  const double minv = phi.minCoeff();
  phi.array() -= minv;

  return phi;
}

Eigen::MatrixXd HeatMethodGeodesics::distance_matrix(const std::vector<int>& sources) const {
  Eigen::MatrixXd d(mesh_.n_vertices(), static_cast<Eigen::Index>(sources.size()));
  for (Eigen::Index j = 0; j < d.cols(); ++j) {
    d.col(j) = distance_from(sources[static_cast<std::size_t>(j)]);
  }
  return d;
}

}  // namespace spectral_bandit
