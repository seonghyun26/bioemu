cd ../

method=ours
molecule=cln025
date=0831_223050
echo $date

# python main.py \
#     --config-name $method-$molecule \
#     date=$date \
#     start_gpu=4 \
#     +tags=['svr']

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr']
