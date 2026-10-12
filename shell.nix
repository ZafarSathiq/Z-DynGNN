{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "sandcert-dev";
  buildInputs = with pkgs; [
    python311
    uv
    git
    curl
    cacert
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
  '';
}
