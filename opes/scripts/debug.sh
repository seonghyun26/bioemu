cd ../

method="debug"
date="debug"
echo $date

# python main.py \
#     --config-name $method \
#     date=$date \
#     +tags=['debug']

wait 

python analysis_opes.py \
    --config-name $method \
    date=$date 
