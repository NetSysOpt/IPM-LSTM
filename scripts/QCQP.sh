---
#Convex_QCQP_RHS
python main.py --config ./configs/QCQP.yaml --prob_type Convex_QCQP_RHS --ineq_tol 0.001 --eq_tol 0.002
python main.py --config ./configs/QCQP.yaml --prob_type Convex_QCQP_RHS --ineq_tol 0.001 --eq_tol 0.002 --test --test_solver ipopt --save_sol


#Convex_QCQP
python main.py --config ./configs/QCQP.yaml --prob_type Convex_QCQP


