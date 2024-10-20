---

#Nonconvex_Program_RHS 100-50-50
python main.py --config .\configs\Nonconvex_Program.yaml --prob_type Nonconvex_Program_RHS
python main.py --config .\configs\Nonconvex_Program.yaml --prob_type Nonconvex_Program_RHS --test --test_solver ipopt

#Nonconvex_Program_RHS 200-100-100
python main.py --config .\configs\Nonconvex_Program.yaml --num_var 200 --num_eq 100 --num_ineq 100 --prob_type Nonconvex_Program_RHS --hidden_dim 75
python main.py --config .\configs\Nonconvex_Program.yaml --num_var 200 --num_eq 100 --num_ineq 100 --prob_type Nonconvex_Program_RHS --hidden_dim 75 --test --test_solver ipopt --save_sol


#Nonconvex_Program 100-50-50
python main.py --config .\configs\Nonconvex_Program.yaml --prob_type Nonconvex_Program
python main.py --config .\configs\Nonconvex_Program.yaml --prob_type Nonconvex_Program --test --test_solver ipopt

#Nonconvex_Program 200-100-100
python main.py --config .\configs\Nonconvex_Program.yaml --num_var 200 --num_eq 100 --num_ineq 100 --prob_type Nonconvex_Program --hidden_dim 100
python main.py --config .\configs\Nonconvex_Program.yaml --num_var 200 --num_eq 100 --num_ineq 100 --prob_type Nonconvex_Program --hidden_dim 100 --test --test_solver ipopt --save_sol

