"""
Ice skating blade physics server — modular package.

Modules:
  config          Constants and simulation parameters
  kernels         Warp GPU kernels (particle init, physics steps, reduction)
  blade_mesh      STL mesh loading and transformation
  blade_geometry  Rocker profile, hollow-grind cross-section, contact lookup table
  physics         BladePhysics simulation class (state, settling, stepping, commands)
  web             HTTP routes, WebSocket handler, physics loop
"""
import warp as wp

wp.init()
wp.set_device("cuda:0")
