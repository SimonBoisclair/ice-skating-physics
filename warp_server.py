"""
Entry point — delegates to server package.

    python3 warp_server.py

See server/ for the modular source:
  config.py          Constants
  kernels.py         GPU kernels
  blade_mesh.py      STL loading
  blade_geometry.py  Contact geometry
  physics.py         Simulation
  web.py             HTTP + WebSocket
"""
from server.web import run_server

if __name__ == '__main__':
    run_server()
