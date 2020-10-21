python _init_fresque.py
echo "Applying Python program on all txt files"
for f in resultats/*.txt; do
    python __ransac.py $f
    echo "$f is done."
done