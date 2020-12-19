#!/bin/bash
# Partie 1 : creer les fichiers txt
echo "Starting execution of deepmatch for all images "
let numero=1
#for f in images/frag/*.ppm;
for f in images/frag/*.ppm 
do
    echo "-> $f"
    ./deepmatching-static $f images/fresque.ppm -rot_range 0 359 -max_scale 1 -nt 30 -out resultats3/$numero.txt
    let numero++
done