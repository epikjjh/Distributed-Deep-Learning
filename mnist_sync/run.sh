# parameter server: 1
# worker: 2
mpiexec -n $1 python3 parameter_server.py : -n $2 python3 worker.py 
