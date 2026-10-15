{
  description = "Z-DynGNN: Z-score Calibrated Dynamics-Aware Graph Neural Network";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
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
            echo "[+] Z-DynGNN Nix Flake environment active (Python $(python --version 2>&1), uv $(uv --version 2>&1))"
          '';
        };
      });
    };
}
