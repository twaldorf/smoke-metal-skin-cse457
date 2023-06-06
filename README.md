# smoke-metal-skin-cse457
This project extends the RayTracingInOneWeekend Book 1 path tracer with CPU multithreading, CUDA and OptiX support, and new geometry and materials such as cubes, emissives, and volumes. Created by Group K: Advanced Raytracing, UW CSE457 SP23

Requires Boost, libpng16, for CPU backend.

Requires stb_image, g++12 (nvcc 12.x is not currently compatible with g++13), CUDA 12.x, and OptiX 7.x for the two GPU backends.

Edit `set(OptiX_INSTALL_DIR /opt/optix)` in `CMakeLists.txt:37` to point to your OptiX install directory.

Currently resolution and sample count are hardcoded but they can easily be changed by editing the `screenInfo` values at the top of `main.cpp` and rebuilding.

# Building
 1. Ensure appropriate dependencies are installed
 2. Create and `cd` into a build directory (`mkdir build && cd build`)
 3. Run `cmake ..`
 
# Running 
## CPU
Run `./rtiow1`
## GPU 
### CUDA
Run `./rtiow1 --gpu`
### Optix
Run `./rtiow1 --optix`
