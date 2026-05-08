#!/bin/bash
DIR=/workspace/ice-skating-physics
LOG=/workspace/server.log

restart_server() {
    echo "[watch] $(date '+%H:%M:%S') File changed: $1 — restarting server..."
    for pid in $(pgrep -f 'python warp_server.py'); do
        if [ "$pid" != "1" ] && [ "$pid" != "$$" ]; then
            kill $pid 2>/dev/null
        fi
    done
    sleep 2
    cd $DIR
    PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 nohup python warp_server.py > $LOG 2>&1 &
    echo "[watch] Server restarted (PID $!)"
}

echo "[watch] Watching .py .html .js .css files in $DIR for changes..."
echo "[watch] Press Ctrl+C to stop"

inotifywait -m -r -e modify,create --include '.*\.(py|html|js|css)$' $DIR | while read path action file; do
    echo "[watch] Detected $action on $path$file"
    restart_server "$path$file"
    sleep 2
    while read -t 0.1 _ _ _; do :; done
done
