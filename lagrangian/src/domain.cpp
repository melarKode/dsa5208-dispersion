#include "lps/domain.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lps {

namespace {

// Wraps `coord` into [origin, origin + length) and adds the number of
// periods shifted to `image`, so coord + image * length is unchanged.
void wrap_periodic(double& coord, std::int32_t& image, double origin, double length) {
  double s = coord - origin;
  const double periods = std::floor(s / length);
  s -= periods * length;
  std::int32_t shift = static_cast<std::int32_t>(periods);
  // Round-off can leave s == length (e.g. a tiny negative s plus length).
  if (s >= length) {
    s -= length;
    shift += 1;
  } else if (s < 0.0) {
    s += length;
    shift -= 1;
  }
  coord = origin + s;
  image += shift;
}

bool inside_open_axis(double coord, double origin, double length) {
  return coord >= origin && coord < origin + length;
}

}  // namespace

Domain::Domain(const Grid& grid, const std::array<BoundaryType, 3>& boundary, std::vector<unsigned char> mask)
    : mask_(std::move(mask)), grid_(grid), boundary_(boundary) {
  if(!mask_.empty() && mask_.size()!=static_cast<std::size_t>(grid.num_cells())) throw std::runtime_error("Mask size mismatch");
}

double Domain::length(int axis) const {
  switch (axis) {
    case 0: return grid_.length_x();
    case 1: return grid_.length_y();
    default: return grid_.length_z();
  }
}

StepOutcome Domain::apply_boundaries(Vec3& position, std::int32_t image[3]) const {
  double* coords[3] = {&position.x, &position.y, &position.z};
  const double origins[3] = {grid_.origin_x, grid_.origin_y, grid_.origin_z};
  for (int axis = 0; axis < 3; ++axis) {
    if (boundary_[axis] == BoundaryType::Periodic) {
      wrap_periodic(*coords[axis], image[axis], origins[axis], length(axis));
    } else if (boundary_[axis] == BoundaryType::Closed) {
      double q=std::fmod(*coords[axis]-origins[axis],2*length(axis));
      if(q<0) q+=2*length(axis);
      if(q>=length(axis)) q=2*length(axis)-q;
      *coords[axis]=std::min(origins[axis]+q,std::nextafter(origins[axis]+length(axis),origins[axis]));
    } else if (!inside_open_axis(*coords[axis], origins[axis], length(axis))) {
      return StepOutcome::Escaped;
    }
  }
  return StepOutcome::Active;
}

StepOutcome Domain::resolve_step(const Vec3& start, Vec3& end, std::int32_t image[3], bool reflect_open) const {
  if(mask_.empty() && !reflect_open && std::none_of(boundary_.begin(),boundary_.end(),
      [](BoundaryType b){return b==BoundaryType::Closed;})) return apply_boundaries(end,image);
  double p[]={start.x,start.y,start.z}, delta[]={end.x-start.x,end.y-start.y,end.z-start.z};
  const double o[]={grid_.origin_x,grid_.origin_y,grid_.origin_z}, h[]={grid_.dx,grid_.dy,grid_.dz};
  const int n[]={grid_.nx,grid_.ny,grid_.nz};
  int cell[3];
  if(!grid_.locate(start,cell[0],cell[1],cell[2]) || is_solid_cell(grid_.cell_id(cell[0],cell[1],cell[2])))
    throw std::runtime_error("Particle segment starts outside air");
  for(int crossing=0;crossing<100000;++crossing) {
    double fraction=2;int axis=-1,sign=0;
    for(int a=0;a<3;++a) if(delta[a]!=0) {
      int direction=delta[a]>0?1:-1;
      double face=o[a]+(cell[a]+(direction>0))*h[a];
      double t=(face-p[a])/delta[a];
      if(t>=-1e-10 && t<=1 && t<fraction) {fraction=std::max(t,0.);axis=a;sign=direction;}
    }
    if(axis<0) {
      end={p[0]+delta[0],p[1]+delta[1],p[2]+delta[2]};
      int i,j,k;
      if(!grid_.locate(end,i,j,k) || is_solid_cell(grid_.cell_id(i,j,k))) throw std::runtime_error("Invalid particle segment endpoint");
      return StepOutcome::Active;
    }
    for(int a=0;a<3;++a) {p[a]+=fraction*delta[a];delta[a]*=1-fraction;}
    int next[]={cell[0],cell[1],cell[2]};next[axis]+=sign;
    bool outside=next[axis]<0 || next[axis]>=n[axis];
    bool reflect=outside && (boundary_[axis]==BoundaryType::Closed || (reflect_open && boundary_[axis]==BoundaryType::Open));
    if(outside && !reflect && boundary_[axis]==BoundaryType::Open) return StepOutcome::Escaped;
    if(outside && boundary_[axis]==BoundaryType::Periodic) next[axis]=(next[axis]+n[axis])%n[axis];
    if(!reflect) reflect=is_solid_cell(grid_.cell_id(next[0],next[1],next[2]));
    const double epsilon=1e-12*h[axis];
    if(reflect) {delta[axis]=-delta[axis];p[axis]-=sign*epsilon;}
    else {
      if(outside) {p[axis]-=sign*n[axis]*h[axis];image[axis]+=sign;}
      cell[axis]=next[axis];p[axis]+=sign*epsilon;
    }
  }
  throw std::runtime_error("Too many cell crossings; reduce dt");
}

}  // namespace lps
