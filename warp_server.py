"""
Entry point — delegates to server package.

    python3 warp_server.py

See server/ for the modular source:
  config.py          Constants
  kernels.py         GPU particle kernels
  particle_sim.py    Particles-only pool simulation
  renderer_warp.py   CAD + particle renderer
  web.py             HTTP + WebSocket
"""
from server.web import run_server

if __name__ == '__main__':
    run_server()
