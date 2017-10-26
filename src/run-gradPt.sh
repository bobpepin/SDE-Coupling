rm -r output_gradPt/
mkdir output_gradPt
python generate_experiment.py experiment_gradPt.asdf
python simulate_Pt.py output_gradPt/
# for d in ou doublewell; do
#      for k in 0 8; do
#           for s in 0 8; do
#               python compute_Pt.py output_gradPt/XY_dynamics\=$d,kappa_Y=$k,sigma_Y=$s*
#           done
#      done
# done
# python compute_Pt.py output_gradPt/XY_dynamics\=ou,kappa_Y=1,sigma_Y=1*
# python compute_Pt.py output_gradPt/XY_dynamics\=doublewell*

python compute_Pt.py output_gradPt/
