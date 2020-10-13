echo "Starting execution of deepmatch for all images :"
mkdir resultats
let "numero = 1"
for f in images/frag/*.ppm; do
    echo "-> $f"
    ./deepmatching-static $f images/fresque.ppm -out resultats/$numero.txt
    let "numero = numero+1"
    echo "done."
done



