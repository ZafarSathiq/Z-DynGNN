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
  '';
}
