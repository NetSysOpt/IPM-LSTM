---
#Convex_QP_RHS
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Convex_QP_RHS
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Convex_QP_RHS --num_var 200 --num_eq 100 --num_ineq 100

#Nonconvex_Program_RHS
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_Program_RHS
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_Program_RHS --num_var 200 --num_eq 100 --num_ineq 100

# Convex_QCQP_RHS
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Convex_QCQP_RHS
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Convex_QCQP_RHS --num_var 200 --num_eq 100 --num_ineq 100

#Nonconvex_QP
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name qp1
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name qp2
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name st_rv1
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name st_rv2
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name st_rv3
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name st_rv7
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name st_rv9
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Nonconvex_QP --mat_name qp30_15_1_1


