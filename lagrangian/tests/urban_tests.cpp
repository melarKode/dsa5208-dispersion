#include "lps/domain.hpp"
#include "lps/urban.hpp"
#include "lps/particles.hpp"
#include "lps/stepper.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace lps;
void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
void near(double a,double b){check(std::abs(a-b)<1e-8,"Coordinate mismatch");}
int main(int argc,char** argv) {
 try {
  check(argc==2,"Temporary fixture path required");
  Grid g;g.nx=10;g.ny=10;g.nz=10;g.dx=g.dy=g.dz=1;
  std::vector<unsigned char> mask(1000,0);
  for(int k=0;k<10;++k)for(int j=0;j<10;++j)mask[g.cell_id(5,j,k)]=1;
  std::array<BoundaryType,3> closed={BoundaryType::Closed,BoundaryType::Closed,BoundaryType::Closed};
  Domain wall(g,closed,mask);
  std::int32_t image[3]={};Vec3 end{8,2,2};
  check(wall.resolve_step({4,2,2},end,image)==StepOutcome::Active,"Thin wall crossing");near(end.x,2);
  end={-23,2,2};wall.resolve_step({2,2,2},end,image);near(end.x,3);
  end={8,8,8};wall.resolve_step({2,2,2},end,image);check(end.x<5,"Diagonal tunnel");
  Grid shifted=g;shifted.origin_x=-700;shifted.origin_y=-650;shifted.origin_z=1.4;shifted.dx=shifted.dy=2.34;shifted.dz=2;
  Domain shifted_wall(shifted,closed,mask);
  end={-700+8*2.34,-650+2*2.34,5.4};
  shifted_wall.resolve_step({-700+4*2.34,-650+2*2.34,5.4},end,image);near(end.x,-700+2*2.34);
  Domain box(g,closed,std::vector<unsigned char>(1000,0));
  end={32,2,2};box.resolve_step({2,2,2},end,image);near(end.x,8);
  auto open=closed;open[0]=BoundaryType::Open;Domain outlet(g,open,std::vector<unsigned char>(1000,0));
  end={11,2,2};check(outlet.resolve_step({9,2,2},end,image)==StepOutcome::Escaped,"Advection must escape");
  end={11,2,2};check(outlet.resolve_step({9,2,2},end,image,true)==StepOutcome::Active,"Diffusion must reflect");near(end.x,9);
  auto periodic=closed;periodic[0]=BoundaryType::Periodic;Domain wrap(g,periodic,std::vector<unsigned char>(1000,0));
  image[0]=0;end={32,2,2};wrap.resolve_step({2,2,2},end,image);near(end.x,2);check(image[0]==3,"Multiple wraps lost");
  std::mt19937 rng(43);std::normal_distribution<double> normal(0,3);Vec3 pos{2,2,2};
  for(int q=0;q<20000;++q){end={pos.x+normal(rng),pos.y+normal(rng),pos.z+normal(rng)};wall.resolve_step(pos,end,image,true);check(end.x<5,"Random walk tunnel");pos=end;}
  Config c;c.nx=c.ny=c.nz=10;c.dx=c.dy=c.dz=1;c.boundary=closed;c.velocity_mode=VelocityMode::File;c.case_dir=argv[1];
  std::filesystem::create_directories(c.case_dir);
  {std::ofstream f(c.case_dir+"/solid.bin",std::ios::binary);f.write(reinterpret_cast<char*>(mask.data()),mask.size());}
  for(const auto* name:{"u","v","w"}) {std::vector<double> wind(1000,2);std::ofstream f(c.case_dir+"/"+name+".bin",std::ios::binary);f.write(reinterpret_cast<char*>(wind.data()),wind.size()*sizeof(double));}
  UrbanWind wind(c);near(wind.velocity({2.5,2.5,2.5},0).x,2);near(wind.velocity({4.999999999,2,2},0).x,0);
  near(wind.velocity({0.000000001,2,2},0).x,0);
  c.source_x=4.9;c.source_y=c.source_z=5;c.source_sigma=1;c.particle_count=10000;c.initial_mass=1;c.seed=12;
  auto release=make_gaussian_release(c,wall);check(release.escaped_particles==0,"Conditioned release lost particles");
  for(std::size_t i=0;i<release.particles.size();++i){int x,y,z;check(g.locate(release.particles.position(i),x,y,z),"Release outside");check(!mask[g.cell_id(x,y,z)],"Release inside building");}
  ParticleSet cloud;cloud.resize(20000);std::uniform_real_distribution<double> uniform(0,1);
  for(std::size_t i=0;i<cloud.size();++i) {
    cloud.id[i]=i;cloud.active[i]=1;cloud.mass[i]=1./cloud.size();
    cloud.x[i]=5*uniform(rng);cloud.y[i]=10*uniform(rng);cloud.z[i]=10*uniform(rng);
  }
  UniformWind still({0,0,0});StepParameters params{456,.05,1};
  for(unsigned step=0;step<50;++step)check(advance_particles(cloud,still,wall,params,step).escaped_particles==0,"Reflected diffusion escaped");
  int bins[5]={};for(auto x:cloud.x)++bins[static_cast<int>(x)];
  for(int count:bins)check(std::abs(count-4000)<300,"Uniform distribution not preserved by reflection");
  for(int a=0;a<3;++a) {
    std::vector<double> values(1000);const char* names[]={"u","v","w"};
    for(int k=0;k<10;++k)for(int j=0;j<10;++j)for(int i=0;i<10;++i)values[g.cell_id(i,j,k)]=(a==0?i:(a==1?j:k))+.5;
    std::ofstream f(c.case_dir+"/"+names[a]+".bin",std::ios::binary);f.write(reinterpret_cast<char*>(values.data()),values.size()*sizeof(double));
  }
  UrbanWind linear(c);auto v=linear.velocity({2.2,3.3,4.4},0);near(v.x,2.2);near(v.y,3.3);near(v.z,4.4);
  std::cout<<"PASS: thin walls, repeated reflection, diagonal collisions, open split, periodic wraps, 20000 random paths, wall-normal/linear wind, conditioned release, stationary uniform diffusion\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
