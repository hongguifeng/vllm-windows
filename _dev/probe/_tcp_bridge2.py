#!/usr/bin/env python3
"""TCP bridge: expose a WSL-local endpoint to docker bridge containers.

Usage: _tcp_bridge2.py <listen_port> <target_host> <target_port>
"""

import socket
import sys
import threading


def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def handle(client, target):
    upstream = socket.create_connection(target, timeout=10)
    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()


def main():
    listen_port = int(sys.argv[1])
    target = (sys.argv[2], int(sys.argv[3]))
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", listen_port))
    server.listen(128)
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client, target), daemon=True).start()


if __name__ == "__main__":
    main()
