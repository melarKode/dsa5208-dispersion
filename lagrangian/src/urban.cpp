#include "lps/urban.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace lps {
namespace {
template<class T> std::vector<T> read(const Config& c, const std::string& name) {
  const auto path=std::filesystem::path(c.case_dir)/name;
  const auto n=c.grid().num_cells();
  if (n<=0 || std::filesystem::exists(std::filesystem::path(c.case_dir)/"INCOMPLETE"))
    throw std::runtime_error("Invalid or incomplete prepared case");
  if (std::filesystem::file_size(path)!=static_cast<std::uintmax_t>(n)*sizeof(T))
    throw std::runtime_error("Wrong binary size: "+path.string());
  std::vector<T> data(static_cast<std::size_t>(n));
  std::ifstream in(path,std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(data.data()),n*sizeof(T))) throw std::runtime_error("Binary read failed");
  return data;
}
}
std::vector<unsigned char> load_solid(const Config& c) {
  if(c.velocity_mode!=VelocityMode::File) return {};
  auto mask=read<unsigned char>(c,"solid.bin");
  for(auto m:mask) if(m>1) throw std::runtime_error("Nonbinary solid mask");
  return mask;
}
UrbanWind::UrbanWind(const Config& c):grid_(c.grid()),boundary_(c.boundary),mask_(load_solid(c)) {
  const unsigned short endian=1;
  if(*reinterpret_cast<const unsigned char*>(&endian)!=1) throw std::runtime_error("Little-endian host required");
  const char* names[]={"u.bin","v.bin","w.bin"};
  for(int a=0;a<3;++a) {
    wind_[a]=read<double>(c,names[a]);
    for(auto v:wind_[a]) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite wind");
  }
}
Vec3 UrbanWind::velocity(const Vec3& p,double) const {
  int cell[3];
  if(!grid_.locate(p,cell[0],cell[1],cell[2])) throw std::runtime_error("Wind queried outside domain");
  auto id=grid_.cell_id(cell[0],cell[1],cell[2]);
  if(mask_[id]) throw std::runtime_error("Wind queried inside solid");
  int n[]={grid_.nx,grid_.ny,grid_.nz};
  double h[]={grid_.dx,grid_.dy,grid_.dz},o[]={grid_.origin_x,grid_.origin_y,grid_.origin_z};
  double pos[]={p.x,p.y,p.z},v[3];
  for(int a=0;a<3;++a) {
    double face[2];
    for(int side=0;side<2;++side) {
      int other[]={cell[0],cell[1],cell[2]};other[a]+=side?1:-1;
      if(other[a]<0 || other[a]>=n[a]) {
        if(boundary_[a]!=BoundaryType::Periodic) {
          face[side]=boundary_[a]==BoundaryType::Closed?0:wind_[a][id];continue;
        }
        other[a]=(other[a]+n[a])%n[a];
      }
      auto nb=grid_.cell_id(other[0],other[1],other[2]);
      face[side]=mask_[nb]?0:.5*(wind_[a][id]+wind_[a][nb]);
    }
    double f=(pos[a]-(o[a]+cell[a]*h[a]))/h[a];
    v[a]=(1-f)*face[0]+f*face[1];
  }
  return {v[0],v[1],v[2]};
}
}
