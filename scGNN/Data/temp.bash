python Preprocessing_benchmark.py \
    --inputfile Data/benchmarkData/Chung/T2000_expression.txt \
    --outputfile Data/Chung.csv \
    --split space --cellheadflag False --cellcount 317

python Preprocessing_main.py \
    --expression-name Chung \
    --featureDir Data/

python3 -W ignore main_benchmark.py \
    --datasetName Chung \
    --benchmark Data/benchmarkData/Chung/Chung_cell_label.csv \
    --LTMGDir Data/benchmarkData/ \
    --regulized-type LTMG \
    --EMtype celltypeEM \
    --clustering-method LouvainK \
    --useGAEembedding \
    --npyDir output/ \
    --debuginfo 