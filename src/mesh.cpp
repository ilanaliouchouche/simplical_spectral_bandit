#include "spectral_bandit/mesh.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace spectral_bandit {
namespace {

std::string trim(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
    ++b;
  }
  std::size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
    --e;
  }
  return s.substr(b, e - b);
}

int parse_vertex_index_0based(const std::string& tok) {
  auto slash = tok.find('/');
  std::string head = (slash == std::string::npos) ? tok : tok.substr(0, slash);
  if (head.empty()) {
    throw std::runtime_error("Invalid OBJ face token: '" + tok + "'");
  }

  const int idx1 = std::stoi(head);
  if (idx1 == 0) {
    throw std::runtime_error("OBJ indexing starts at 1 (got 0)");
  }
  if (idx1 < 0) {
    throw std::runtime_error("Negative OBJ indices are not supported in this loader");
  }
  return idx1 - 1;
}

}  // namespace

Mesh load_obj(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open OBJ: " + path.string());
  }

  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3i> faces;

  std::string raw;
  while (std::getline(in, raw)) {
    const std::string line = trim(raw);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    if (line.rfind("v ", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      iss >> tag >> x >> y >> z;
      if (!iss.fail()) {
        vertices.emplace_back(x, y, z);
      }
      continue;
    }

    if (line.rfind("f ", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      iss >> tag;

      std::vector<int> poly;
      std::string tok;
      while (iss >> tok) {
        poly.push_back(parse_vertex_index_0based(tok));
      }

      if (poly.size() < 3) {
        continue;
      }

      const int root = poly[0];
      for (std::size_t i = 1; i + 1 < poly.size(); ++i) {
        faces.emplace_back(root, poly[i], poly[i + 1]);
      }
    }
  }

  if (vertices.empty() || faces.empty()) {
    throw std::runtime_error("Failed to load mesh from " + path.string() + ": missing vertices or faces");
  }

  Mesh mesh;
  mesh.vertices.resize(static_cast<Eigen::Index>(vertices.size()), 3);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(vertices.size()); ++i) {
    mesh.vertices.row(i) = vertices[static_cast<std::size_t>(i)].transpose();
  }

  mesh.faces.resize(static_cast<Eigen::Index>(faces.size()), 3);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(faces.size()); ++i) {
    const auto& f = faces[static_cast<std::size_t>(i)];
    if (f.minCoeff() < 0 || f.maxCoeff() >= mesh.vertices.rows()) {
      throw std::runtime_error("Face indices out of range in " + path.string());
    }
    mesh.faces.row(i) = f.transpose();
  }

  return mesh;
}

}  // namespace spectral_bandit
