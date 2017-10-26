P="$@"
echo Prefix: $P
python generate_experiment.py experiment.asdf && python sde_eigen.py ${P}sample_* && python I_var.py ${P}I_XY_* > ${P}I_var.csv
