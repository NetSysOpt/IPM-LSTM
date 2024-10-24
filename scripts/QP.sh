---

#Convex QP
python main.py --config ./configs/QP.yaml --prob_type QP_RHS
python main.py --config ./configs/QP.yaml --prob_type QP_RHS --test --solver ipopt --save_sol

python main.py --config ./configs/QP.yaml --prob_type QP_RHS --num_var 200 --num_ineq 100 --num_eq 100 --hidden_dim 75
python main.py --config ./configs/QP.yaml --prob_type QP_RHS --num_var 200 --num_ineq 100 --num_eq 100 --hidden_dim 75 --test --solver ipopt --save_sol


python main.py --config ./configs/QP.yaml --prob_type QP
python main.py --config ./configs/QP.yaml --prob_type QP --test --solver ipopt --save_sol


#Nonconvex QP
python main.py --config ./configs/QP.yaml --prob_type Nonconvex_QP --mat_name st_rv7
python main.py --config ./configs/QP.yaml --prob_type Nonconvex_QP --mat_name st_rv7 --test --test_solver ipopt
