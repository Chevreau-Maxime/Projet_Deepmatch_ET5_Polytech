python __init_fresque.py
echo "Applying Python program on all txt files"
for f in resultats3/*.txt; do
    python __main.py $f
    echo "$f is done."
done