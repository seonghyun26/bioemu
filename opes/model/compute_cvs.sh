# CLN025
# 2JOF
# 2F4K

for molecule in 1FME; do
    python all_cv.py --method tda --molecule $molecule
    python all_cv.py --method tae --molecule $molecule
    python all_cv.py --method tica --molecule $molecule
    python all_cv.py --method vde --molecule $molecule
done

