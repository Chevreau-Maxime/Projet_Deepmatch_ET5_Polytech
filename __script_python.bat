echo "Go !"
for %%f in (.\resultats3\*) do ( 
    @echo %%f 
    python __ransac.py %%f
)