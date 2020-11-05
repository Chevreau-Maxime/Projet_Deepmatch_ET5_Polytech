# Partie 1 : creer les fichiers txt
echo "Starting execution of deepmatch for all images "
mkdir resultats
let "numero = 1"
for f in images/frag/*.ppm; do
    echo "-> $f"
    ./deepmatching-static $f images/fresque.ppm -rot_range 0 359 -out resultats/$numero.txt
    let "numero = numero+1"
    echo "done."
done
# Partie 2 : traiter les fichiers txt
