python main.py --num_locations 40 --query_source all --comparison population_weighted
sleep 1h

python main.py --num_locations 10 --query_source all --comparison income
sleep 1h

python main.py --num_locations 10 --query_source all --comparison voting
sleep 1h

python main.py --num_locations 10 --query_source all --comparison urban-rural
sleep 1h

python main.py --num_locations 40 --query_source extra --comparison population_weighted
sleep 1h

python main.py --num_locations 10 --query_source extra --comparison income
sleep 1h

python main.py --num_locations 10 --query_source extra --comparison voting
sleep 1h

python main.py --num_locations 10 --query_source extra --comparison urban-rural
sleep 1h
