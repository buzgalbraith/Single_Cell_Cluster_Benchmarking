python ./scGNN/Preprocessing_benchmark.py \
    --inputfile data/benchmark_data/Chung/T2000_expression.txt \
    --outputfile data/benchmark_data/Chung/Chung.csv \
    --split space \
    --cellheadflag False \
    --cellcount 317
python ./scGNN/Preprocessing_benchmark.py \
    --inputfile data/benchmark_data/Kolodziejczyk/T2000_expression.txt \
    --outputfile data/benchmark_data/Kolodziejczyk/Kolodziejczyk.csv \
    --split space \
    --cellheadflag False \
    --cellcount 704

python ./scGNN/Preprocessing_benchmark.py \
    --inputfile data/benchmark_data/Klein/T2000_expression.txt \
    --outputfile data/benchmark_data/Klein/Klein.csv \
    --split space \
    --cellheadflag False \
    --cellcount 2717

python ./scGNN/Preprocessing_benchmark.py \
    --inputfile data/benchmark_data/Zeisel/T2000_expression.txt \
    --outputfile data/benchmark_data/Zeisel/Zeisel.csv \
    --split space \
    --cellheadflag False \
    --cellcount 3005