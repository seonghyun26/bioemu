cd ../

step=600_000_000

python main.py \
    --config-name tda \
    gpu=4 \
    step=$step &

sleep 1

python main.py \
    --config-name tae \
    gpu=5 \
    step=$step &

sleep 1

python main.py \
    --config-name tica \
    gpu=6 \
    step=$step &

sleep 1

python main.py \
    --config-name vde \
    gpu=7 \
    step=$step &