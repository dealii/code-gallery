
#-----------------------------------------------------------------------------------------
#
# This python script runs the deal.ii verification problems based on Ehlers & Eipper 1999
# To execute, type in console >>"python ./run-multi-calc.py"
#
#   *** remember to load deal.ii and mpi before executing the script***
#
# NOTE: If changes are made to the file runPoro.sh, remember to define it as executable
#       by typing in the console >>"chmod +x runPoro.sh"
#
#-----------------------------------------------------------------------------------------
import fileinput
import sys
import subprocess
import os
import shutil
import select
import time

# Number of parallel processes, i.e. how many jobs are run in parallel.
# Since the deal.ii code is executed already in parallel (-np 6), it doesn't make sense
# to execute any jobs in parallel. So , parallelProcesses = 1 should be used.
parallelProcesses = 1
processes = {}
jobnames = {}


# Generate new parameter file
x = open("parameters.prm", 'w')
x.write("subsection Finite element system\n")
x.write("  set Polynomial degree displ = 2\n")
x.write("  set Polynomial degree pore  = 1\n")
x.write("  set Quadrature order        = 3\n")
x.write("end\n\n")
x.write("subsection Geometry\n")
x.write("  set Geometry type       = Ehlers_tube_step_load\n")
x.write("  set Global refinement   = 1\n")
x.write("  set Grid scale          = 1\n")
x.write("  set Load type           = pressure\n")
x.write("  set Load value          = -7.5e6\n")
x.write("  set Drained pressure    = 0\n")
x.write("end\n\n")
x.write("subsection Material properties \n")
x.write("  set material                         = Neo-Hooke\n")
x.write("  set lambda                           = 8.375e6\n")
x.write("  set shear modulus                    = 5.583e6\n")
x.write("  set seepage definition               = Ehlers\n")
x.write("  set initial solid volume fraction    = 0.67\n")
x.write("  set kappa                            = 0\n")
x.write("  set initial Darcy coefficient        = 1.0e-4\n")
x.write("  set fluid weight                     = 1.0e4\n")
x.write("  set gravity term                     = false\n")
x.write("end\n\n")
x.write("subsection Nonlinear solver\n")
x.write("  set Max iterations Newton-Raphson = 20\n")
x.write("  set Tolerance force               = 1.0e-6\n")
x.write("  set Tolerance displacement        = 1.0e-6\n")
x.write("  set Tolerance pore pressure       = 1.0e-6\n")
x.write("end\n\n")
x.write("subsection Time\n")
x.write("  set End time          = 10.0\n")
x.write("  set Time step size    = 0.002\n")
x.write("end\n\n")
x.write("subsection Output parameters\n")
x.write("  set Time step number output    = 5\n")
x.write("end\n")
x.close()

#---------------------------------------------------------------------------------------
# Step loading problem with different kappa values
#---------------------------------------------------------------------------------------
kappas = ["0","1","2","5"]

for kappa in kappas:
	# Jobname
	jobname = "Ehlers_step_load_kappa_"+kappa
	# make changes in parameter file
	x = fileinput.input("parameters.prm", inplace=1)
	for line in x:
		if "set kappa" in line:
			line = "  set kappa                            = "+kappa+"\n"
		print (line),
	x.close()

	# start cp fem code without output
	#process = subprocess.Popen("./runPoro.sh " + jobname + "> /dev/null 2>&1", shell=True)
	process = subprocess.Popen("./runPoro.sh " + jobname, shell=True)

	# check if Input folder is copied
	# to make sure I look for the executable which is copied afterwards
	executable = "RESULTS/calcDir_" + jobname
	results = "RESULTS/resultDir_" + jobname
	time.sleep(1)
	while not os.path.exists(executable) and not os.path.exists(results):
		time.sleep(1)
		print ("waiting for executable to be copied")

	# store process to wait for it later
	processes[process.pid] = process
	jobnames[process.pid] = jobname

	# if more than parallelProcesses running, wait for the first to finish
	while len(processes) >= parallelProcesses:
		pid, status = os.wait()
		if pid in processes:
			if status == 0:
				print ("Job %30s successful" % jobnames[pid])
			del processes[pid]
			del jobnames[pid]


# wait for the other processes
while processes:
	pid, status = os.wait()
	if pid in processes:
		if status == 0:
			print ("Job %30s successful" % jobnames[pid])
		del processes[pid]
		del jobnames[pid]

#---------------------------------------------------------------------------------------
# Load increasing problem with kappa = 0
#---------------------------------------------------------------------------------------
# Jobname
jobname = "Ehlers_increase_load_kappa_0"
# make changes in parameter file
x = fileinput.input("parameters.prm", inplace=1)
for line in x:
	if "set Geometry type" in line:
		line = "  set Geometry type       = Ehlers_tube_increase_load\n"
	if "set kappa" in line:
		line = "  set kappa                            = 0\n"
	print (line),
x.close()
# start cp fem code without output
#process = subprocess.Popen("./runPoro.sh " + jobname + "> /dev/null 2>&1", shell=True)
process = subprocess.Popen("./runPoro.sh " + jobname, shell=True)

# check if Input folder is copied
# to make sure I look for the executable which is copied afterwards
executable = "RESULTS/calcDir_" + jobname
results = "RESULTS/resultDir_" + jobname
time.sleep(1)
while not os.path.exists(executable) and not os.path.exists(results):
	time.sleep(1)
	print ("waiting for executable to be copied")

# store process to wait for it later
processes[process.pid] = process
jobnames[process.pid] = jobname

# if more than parallelProcesses running, wait for the first to finish
while len(processes) >= parallelProcesses:
	pid, status = os.wait()
	if pid in processes:
		if status == 0:
			print ("Job %30s successful" % jobnames[pid])
		del processes[pid]
		del jobnames[pid]


#---------------------------------------------------------------------------------------
# Consolidation cube problem
#---------------------------------------------------------------------------------------
# Jobname
jobname = "Ehlers_cube"
# make changes in parameter file
x = fileinput.input("parameters.prm", inplace=1)
for line in x:
	if "set Geometry type" in line:
		line = "  set Geometry type       = Ehlers_cube_consolidation\n"
	if "set Global refinement" in line:
		line = "  set Global refinement   = 2\n"
	if "set Time step size" in line:
		line = "  set Time step size    = 0.01\n"
	if "set Time step number output" in line:
		line = "  set Time step number output    = 1\n"
	print (line),
x.close()

# start cp fem code without output
#process = subprocess.Popen("./runPoro.sh " + jobname + "> /dev/null 2>&1", shell=True)
process = subprocess.Popen("./runPoro.sh " + jobname, shell=True)

# check if Input folder is copied
# to make sure I look for the executable which is copied afterwards
executable = "RESULTS/calcDir_" + jobname
results = "RESULTS/resultDir_" + jobname
time.sleep(1)
while not os.path.exists(executable) and not os.path.exists(results):
	time.sleep(1)
	print ("waiting for executable to be copied")

# store process to wait for it later
processes[process.pid] = process
jobnames[process.pid] = jobname

# if more than parallelProcesses running, wait for the first to finish
while len(processes) >= parallelProcesses:
	pid, status = os.wait()
	if pid in processes:
		if status == 0:
			print ("Job %30s successful" % jobnames[pid])
		del processes[pid]
		del jobnames[pid]


print ("   _   _ _     _     _                           _     _          _ _ ")
print ("  /_\ | | |   (_)___| |__ ___  __ ___ _ __  _ __| |___| |_ ___ __| | |")
print (" / _ \| | |   | / _ \ '_ (_-< / _/ _ \ '  \| '_ \ / -_)  _/ -_) _  |_|")
print ("/_/ \_\_|_|  _/ \___/_.__/__/ \__\___/_|_|_| .__/_\___|\__\___\__,_(_)")
print ("            |__/                           |_|                        ")
