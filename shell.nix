{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "zdyngnn-dev";
  buildInputs = with pkgs; [
    python311
    uv
    git
    curl
    cacert
    zlib
  ];
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
    if [ ! -d ".venv" ]; then
      echo "[*] Creating hermetic Python virtual environment with uv..."
      uv venv .venv --python ${pkgs.python311}/bin/python3
    fi
    source .venv/bin/activate
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    echo "[+] Z-DynGNN classic Nix shell active (Python $(python --version 2>&1), uv $(uv --version 2>&1))"
  '';
}
