
molecule=$1
methods=("tae" "tda" "tica" "vde")
input="$1.dat"

for method in "${methods[@]}"; do
    cp "$input" "../${method}-$1.dat"
    echo "Created ${method}-${molecule}.dat"
done
``

input="${molecule}.yaml"
for method in "${methods[@]}"; do
    output="../${method}-${molecule}.yaml"
    sed "s/methodname/${method}/g" "$input" > "$output"
    echo "Created ${method}-${molecule}.yaml"
done