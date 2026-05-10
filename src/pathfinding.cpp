#include "spectral_bandit/pathfinding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace spectral_bandit {

std::pair<Eigen::VectorXi, double> dijkstra_shortest_path(const AdjacencyList& adjacency, int start, int goal) {
  const int n = static_cast<int>(adjacency.size());
  if (start < 0 || start >= n || goal < 0 || goal >= n) {
    throw std::out_of_range("dijkstra_shortest_path: start/goal out of range");
  }

  if (start == goal) {
    Eigen::VectorXi path(1);
    path(0) = start;
    return {path, 0.0};
  }

  Eigen::VectorXd dist = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
  Eigen::VectorXi prev = Eigen::VectorXi::Constant(n, -1);
  dist(start) = 0.0;

  using QItem = std::pair<double, int>;
  auto cmp = [](const QItem& a, const QItem& b) { return a.first > b.first; };
  std::priority_queue<QItem, std::vector<QItem>, decltype(cmp)> pq(cmp);
  pq.emplace(0.0, start);

  while (!pq.empty()) {
    const auto [cur_d, u] = pq.top();
    pq.pop();

    if (cur_d > dist(u)) {
      continue;
    }
    if (u == goal) {
      break;
    }

    for (const auto& [v, w] : adjacency[static_cast<std::size_t>(u)]) {
      const double nd = cur_d + w;
      if (nd < dist(v)) {
        dist(v) = nd;
        prev(v) = u;
        pq.emplace(nd, v);
      }
    }
  }

  if (!std::isfinite(dist(goal))) {
    return {Eigen::VectorXi(), std::numeric_limits<double>::infinity()};
  }

  std::vector<int> rev;
  for (int u = goal; u != -1; u = prev(u)) {
    rev.push_back(u);
    if (u == start) {
      break;
    }
  }
  std::reverse(rev.begin(), rev.end());

  Eigen::VectorXi path(static_cast<Eigen::Index>(rev.size()));
  for (Eigen::Index i = 0; i < path.size(); ++i) {
    path(i) = rev[static_cast<std::size_t>(i)];
  }

  return {path, dist(goal)};
}

Eigen::VectorXd dijkstra_distances(const AdjacencyList& adjacency, int start) {
  const int n = static_cast<int>(adjacency.size());
  if (start < 0 || start >= n) {
    throw std::out_of_range("dijkstra_distances: start out of range");
  }

  Eigen::VectorXd dist = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
  dist(start) = 0.0;

  using QItem = std::pair<double, int>;
  auto cmp = [](const QItem& a, const QItem& b) { return a.first > b.first; };
  std::priority_queue<QItem, std::vector<QItem>, decltype(cmp)> pq(cmp);
  pq.emplace(0.0, start);

  while (!pq.empty()) {
    const auto [cur_d, u] = pq.top();
    pq.pop();

    if (cur_d > dist(u)) {
      continue;
    }

    for (const auto& [v, w] : adjacency[static_cast<std::size_t>(u)]) {
      const double nd = cur_d + w;
      if (nd < dist(v)) {
        dist(v) = nd;
        pq.emplace(nd, v);
      }
    }
  }

  return dist;
}

}  // namespace spectral_bandit
