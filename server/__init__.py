"""
Particle pool CAD visualization server — modular package.

Modules:
  config          Constants and simulation parameters
  kernels         Warp GPU kernels for particle initialization and DEM contact
  particle_sim    Particles-only pool simulation state and commands
  renderer_warp   GPU CAD and particle stream renderer
  web             HTTP routes, WebSocket handler, physics loop
"""
import warp as wp

wp.init()
wp.set_device("cuda:0")
