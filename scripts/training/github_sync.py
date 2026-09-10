#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Git through a temporary localhost tunnel when cloud DNS is unavailable.

The caller supplies github.com's freshly resolved IPv4 address. Git still uses
https://github.com and verifies its TLS certificate. No global Git/DNS settings
are changed. Can be streamed over SSH before the training checkout exists.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import os
import select
import socket
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", type=ipaddress.IPv4Address, required=True)
    parser.add_argument("git_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    git_args = args.git_args
    if git_args and git_args[0] == "--":
        git_args = git_args[1:]
    if not git_args:
        parser.error("Pass a Git command after --")

    class Tunnel(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *unused):
            pass

        def do_CONNECT(self):
            if self.path != "github.com:443":
                self.send_error(403, "Only github.com:443 is supported")
                return
            try:
                with socket.create_connection((str(args.ip), 443), timeout=15) as upstream:
                    self.send_response(200, "Connection established")
                    self.end_headers()
                    self.wfile.flush()
                    self.close_connection = True
                    while True:
                        readable, _, _ = select.select([self.connection, upstream], [], [], 120)
                        if not readable:
                            return
                        for connection in readable:
                            data = connection.recv(65536)
                            if not data:
                                return
                            destination = (
                                upstream if connection is self.connection else self.connection
                            )
                            destination.sendall(data)
            except OSError:
                return

    environment = dict(
        os.environ, NO_PROXY="", no_proxy="", GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0"
    )
    environment.pop("GIT_SSL_NO_VERIFY", None)
    with ThreadingHTTPServer(("127.0.0.1", 0), Tunnel) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"http.proxy=http://127.0.0.1:{server.server_port}",
                    "-c",
                    "http.sslVerify=true",
                    *git_args,
                ],
                env=environment,
            )
        finally:
            server.shutdown()
            thread.join()
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
