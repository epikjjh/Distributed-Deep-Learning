# parameter server: 1
# worker: 2
mpiexec -n $1 python3 parameter_server.py -np $1 : -n $2 python3 worker.py -np $2
