#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

int threadsPerBlock = 128;

double G = 6.674*std::pow(10,-11);
//double G = 1;

struct simulation {
  size_t nbpart;
  
  std::vector<double> mass;

  //position
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  //velocity
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  //force
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;

  // We need to pass mem to the device aka the gpu, so we can't use the vector directly
  // We need to pass a mem ref essentially. Whenever we pass data to the gpu we cast to a void **, since it is a ptr of a ptr
  double *d_mass = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;
  double *d_z = nullptr;
  double *d_vx = nullptr;
  double *d_vy = nullptr;
  double *d_vz = nullptr;
  double *d_fx = nullptr;
  double *d_fy = nullptr;
  double *d_fz = nullptr;
  
  simulation(size_t nb)
    :nbpart(nb), mass(nb),
     x(nb), y(nb), z(nb),
     vx(nb), vy(nb), vz(nb),
     fx(nb), fy(nb), fz(nb) 
  {}
};

// ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
void CUDA_VERIF(cudaError_t err) {
  if (err != cudaSuccess) { // which is basically 0
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
  }
}

void random_init(simulation& s) {
  std::random_device rd;  
  std::mt19937 gen(rd());
  std::uniform_real_distribution dismass(0.9, 1.);
  std::normal_distribution dispos(0., 1.);
  std::normal_distribution disvel(0., 1.);

  for (size_t i = 0; i<s.nbpart; ++i) {
    s.mass[i] = dismass(gen);

    s.x[i] = dispos(gen);
    s.y[i] = dispos(gen);
    s.z[i] = dispos(gen);
    s.z[i] = 0.;
    
    s.vx[i] = disvel(gen);
    s.vy[i] = disvel(gen);
    s.vz[i] = disvel(gen);
    s.vz[i] = 0.;
    s.vx[i] = s.y[i]*1.5;
    s.vy[i] = -s.x[i]*1.5;
  }

  return;
  //normalize velocity (using normalization found on some physicis blog)
  double meanmass = 0;
  double meanmassvx = 0;
  double meanmassvy = 0;
  double meanmassvz = 0;
  for (size_t i = 0; i<s.nbpart; ++i) {
    meanmass += s.mass[i];
    meanmassvx += s.mass[i] * s.vx[i];
    meanmassvy += s.mass[i] * s.vy[i];
    meanmassvz += s.mass[i] * s.vz[i];
  }
  for (size_t i = 0; i<s.nbpart; ++i) {
    s.vx[i] -= meanmassvx/meanmass;
    s.vy[i] -= meanmassvy/meanmass;
    s.vz[i] -= meanmassvz/meanmass;
  }
  
}

void init_solar(simulation& s) {
  enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
  s = simulation(10);

  // Masses in kg
  s.mass[SUN] = 1.9891 * std::pow(10, 30);
  s.mass[MERCURY] = 3.285 * std::pow(10, 23);
  s.mass[VENUS] = 4.867 * std::pow(10, 24);
  s.mass[EARTH] = 5.972 * std::pow(10, 24);
  s.mass[MARS] = 6.39 * std::pow(10, 23);
  s.mass[JUPITER] = 1.898 * std::pow(10, 27);
  s.mass[SATURN] = 5.683 * std::pow(10, 26);
  s.mass[URANUS] = 8.681 * std::pow(10, 25);
  s.mass[NEPTUNE] = 1.024 * std::pow(10, 26);
  s.mass[MOON] = 7.342 * std::pow(10, 22);

  // Positions (in meters) and velocities (in m/s)
  double AU = 1.496 * std::pow(10, 11); // Astronomical Unit

  s.x = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844*std::pow(10, 8)};
  s.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  s.vx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
  s.vz = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

//meant to update the force that from applies on to
void update_force(simulation& s, size_t from, size_t to) {
  double softening = .1;
  double dist_sq = std::pow(s.x[from]-s.x[to],2)
    + std::pow(s.y[from]-s.y[to],2)
    + std::pow(s.z[from]-s.z[to],2);
  double F = G * s.mass[from]*s.mass[to]/(dist_sq+softening); //that the strength of the force

  //direction
  double dx = s.x[from]-s.x[to];
  double dy = s.y[from]-s.y[to];
  double dz = s.z[from]-s.z[to];
  double norm = std::sqrt(dx*dx+dy*dy+dz*dz);
  
  dx = dx/norm;
  dy = dy/norm;
  dz = dz/norm;

  //apply force
  s.fx[to] += dx*F;
  s.fy[to] += dy*F;
  s.fz[to] += dz*F;
}

void reset_force(simulation& s) {
  for (size_t i=0; i<s.nbpart; ++i) {
    s.fx[i] = 0.;
    s.fy[i] = 0.;
    s.fz[i] = 0.;
  }
}

void apply_force(simulation& s, size_t i, double dt) {
  s.vx[i] += s.fx[i]/s.mass[i]*dt;
  s.vy[i] += s.fy[i]/s.mass[i]*dt;
  s.vz[i] += s.fz[i]/s.mass[i]*dt;
}

void update_position(simulation& s, size_t i, double dt) {
  s.x[i] += s.vx[i]*dt;
  s.y[i] += s.vy[i]*dt;
  s.z[i] += s.vz[i]*dt;
}

void dump_state(simulation& s) {
  std::cout<<s.nbpart<<'\t';
  for (size_t i=0; i<s.nbpart; ++i) {
    std::cout<<s.mass[i]<<'\t';
    std::cout<<s.x[i]<<'\t'<<s.y[i]<<'\t'<<s.z[i]<<'\t';
    std::cout<<s.vx[i]<<'\t'<<s.vy[i]<<'\t'<<s.vz[i]<<'\t';
    std::cout<<s.fx[i]<<'\t'<<s.fy[i]<<'\t'<<s.fz[i]<<'\t';
  }
  std::cout<<'\n';
}

void load_from_file(simulation& s, std::string filename) {
  std::ifstream in (filename);
  size_t nbpart;
  in>>nbpart;
  s = simulation(nbpart);
  for (size_t i=0; i<s.nbpart; ++i) {
    in>>s.mass[i];
    in >>  s.x[i] >>  s.y[i] >>  s.z[i];
    in >> s.vx[i] >> s.vy[i] >> s.vz[i];
    in >> s.fx[i] >> s.fy[i] >> s.fz[i];
  }
  if (!in.good())
    throw "kaboom";
}

void gpu_init(simulation& s) {
  // essentially the number of particles multipled by the data type size (8 bytes)
  size_t N = s.nbpart * sizeof(double);

  CUDA_VERIF(cudaMalloc((void**)&s.d_mass, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_x, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_y, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_z, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_vx, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_vy, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_vz, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_fx, N));
  CUDA_VERIF(cudaMalloc((void**)&s.d_fy, N)); 
  CUDA_VERIF(cudaMalloc((void**)&s.d_fz, N));

  // we need a way to put the calc data from the device to the host data structure
  // std::vectors offer: https://cplusplus.com/reference/vector/vector/data/
  // this gives us the mem ref to the struct
  CUDA_VERIF(cudaMemcpy(s.d_mass, s.mass.data(), N, cudaMemcpyHostToDevice));
  CUDA_VERIF(cudaMemcpy(s.d_x, s.x.data(), N, cudaMemcpyHostToDevice));
  CUDA_VERIF(cudaMemcpy(s.d_y, s.y.data(), N, cudaMemcpyHostToDevice));
  CUDA_VERIF(cudaMemcpy(s.d_z, s.z.data(), N, cudaMemcpyHostToDevice));
  CUDA_VERIF(cudaMemcpy(s.d_vx, s.vx.data(), N, cudaMemcpyHostToDevice));
  CUDA_VERIF(cudaMemcpy(s.d_vy, s.vy.data(), N, cudaMemcpyHostToDevice));
  CUDA_VERIF(cudaMemcpy(s.d_vz, s.vz.data(), N, cudaMemcpyHostToDevice));
}

__global__
void update_forces_gpu(size_t n, double G, double* mass, double* x, double* y, double* z, double* fx, double* fy, double* fz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // the pos using the globally allocated arrays
  // so basically, these are coordinates of particle i
  double xi = x[i];
  double yi = y[i];
  double zi = z[i];
  double fxi = 0;
  double fyi = 0;
  double fzi = 0;
  double softening = 0.1;
  for (size_t j = 0; j < n; ++j) {
    if (i == j) continue;
    double dx = x[j] - xi;
    double dy = y[j] - yi;
    double dz = z[j] - zi;
    double dist_sq = dx*dx + dy*dy + dz*dz;
    double norm = sqrt(dist_sq);

    dx = dx/norm;
    dy = dy/norm;
    dz = dz/norm;

    // force mag
    double F = (G * mass[i] * mass[j]) / (dist_sq + softening);

    fxi += dx*F;
    fyi += dy*F;
    fzi += dz*F;
  }
  fx[i] = fxi;
  fy[i] = fyi;
  fz[i] = fzi;
}

__global__
void update_position_gpu(size_t n, double dt, double* mass, double* vx, double* vy, double* vz, double* fx, double* fy, double* fz, double* x, double* y, double* z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  vx[i] += fx[i] / mass[i] * dt;
  vy[i] += fy[i] / mass[i] * dt;
  vz[i] += fz[i] / mass[i] * dt;
  x[i] += vx[i] * dt;
  y[i] += vy[i] * dt;
  z[i] += vz[i] * dt;
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr
      <<"usage: "<<argv[0]<<" <input> <dt> <nbstep> <printevery> <type>"<<"\n"
      <<"input can be:"<<"\n"
      <<"a number (random initialization)"<<"\n"
      <<"planet (initialize with solar system)"<<"\n"
      <<"a filename (load from file in singleline tsv)"<<"\n";
      return -1;
  }
  
  double dt = std::atof(argv[2]); //in seconds
  size_t nbstep = std::atol(argv[3]);
  size_t printevery = std::atol(argv[4]);
  std::string type = argv[5];

  simulation s(1);

    {
      size_t nbpart = std::atol(argv[1]);
      if ( nbpart > 0) {
        s = simulation(nbpart);
        random_init(s);
      } else {
        std::string inputparam = argv[1];
        if (inputparam == "planet") {
          init_solar(s);
        } else{
          load_from_file(s, inputparam);
        }
      }    
    }

  if (type == "cpu") {  
    for (size_t step = 0; step < nbstep; step++) {
      if (step % printevery == 0)
        dump_state(s);
    
      reset_force(s);
      for (size_t i=0; i<s.nbpart; ++i)
        for (size_t j=0; j<s.nbpart; ++j)
          if (i != j)
            update_force(s, i, j);
      for (size_t i=0; i<s.nbpart; ++i) {
        apply_force(s, i, dt);
        update_position(s, i, dt);
      }
    }  
  } else if (type == "gpu") {
    // offloading to dev
    gpu_init(s);
    int blocks = (s.nbpart + threadsPerBlock - 1) / threadsPerBlock; // ref: https://forums.developer.nvidia.com/t/how-can-i-calculate-blocks-per-grid/248491
    for (size_t step = 0; step < nbstep; step++) {

      update_forces_gpu<<<blocks, threadsPerBlock>>>(s.nbpart, G,
        s.d_mass, s.d_x, s.d_y, s.d_z, s.d_fx, s.d_fy, s.d_fz);

      update_position_gpu<<<blocks, threadsPerBlock>>>(s.nbpart, dt,
        s.d_mass, s.d_vx, s.d_vy, s.d_vz, s.d_fx, s.d_fy, s.d_fz, s.d_x, s.d_y, s.d_z);
      

      CUDA_VERIF(cudaDeviceSynchronize());

      if (step % printevery == 0) {
        // we need to get the current progress, which is why we go w/ device --> host
        CUDA_VERIF(cudaMemcpy(s.x.data(), s.d_x, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.y.data(), s.d_y, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.z.data(), s.d_z, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.vx.data(), s.d_vx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.vy.data(), s.d_vy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.vz.data(), s.d_vz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.fx.data(), s.d_fx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.fy.data(), s.d_fy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_VERIF(cudaMemcpy(s.fz.data(), s.d_fz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost));

        dump_state(s);
      }
    }
  } else {
    std::cout << "Invalid run type\nExpected \"cpu\" or \"gpu\"" << std::endl;
    return 1;
  }

  CUDA_VERIF(cudaFree(s.d_mass));
  CUDA_VERIF(cudaFree(s.d_x));
  CUDA_VERIF(cudaFree(s.d_y));
  CUDA_VERIF(cudaFree(s.d_z));
  CUDA_VERIF(cudaFree(s.d_vx));
  CUDA_VERIF(cudaFree(s.d_vy));
  CUDA_VERIF(cudaFree(s.d_vz));
  CUDA_VERIF(cudaFree(s.d_fx));
  CUDA_VERIF(cudaFree(s.d_fy));
  CUDA_VERIF(cudaFree(s.d_fz));
  return 0;
}
