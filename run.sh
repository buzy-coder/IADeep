# start monitor GPU on each worker node
nvidia-smi dmon -i 0,1,2,3 -d 1 -s ut -o DT -f util_dmon_{hostname}.txt 2>&1 &

# submit tasks
python3 microsoft-job-generartor/submit_tasks.py