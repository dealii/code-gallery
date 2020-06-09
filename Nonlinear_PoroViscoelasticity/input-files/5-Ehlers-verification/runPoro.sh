 #!/bin/bash   
 # first parameter is the name for directory where the calculation is made
 
echo "                    _                _          _   _ _              "
echo " _ _ _  _ _ _  _ _ (_)_ _  __ _   __| |___ __ _| | (_|_)             "
echo "| '_| || | ' \| ' \| | ' \/ _  | / _  / -_) _  | |_| | |  _   _   _  "
echo "|_|  \_,_|_||_|_||_|_|_||_\__, | \__,_\___\__,_|_(_)_|_| (_) (_) (_) "
echo "                          |___/                                      "


echo "DEAL_II_DIR:               " $DEAL_II_DIR

# Define and print tmpdir, where calculations are made and resultsdir where results will be stored
maindir=`pwd`
tmpdir=$maindir/RESULTS/calcDir_$1
resultdir=$maindir/RESULTS/resultDir_$1
echo "Main directory            :" $maindir
echo "Directory for calculations:" $tmpdir
echo "Directory for results     :" $resultdir

# change to temporary job directory
mkdir -p $tmpdir
cd $tmpdir
# copy stuff from location where job was submitted to temporary directory
cp -r $maindir/parameters.prm . && echo "Copying input file succeeded" || echo "Copying input file failed"
cp -r $maindir/nonlinear-poro-viscoelasticity . && echo "Copying executable succeeded" || echo "Copying executable failed"

# run code (change num of parallel processes if desired)
touch out.log
touch err.log
COUNTER=0
while [  $COUNTER -lt 10 ]; do
    COUNTER=$((COUNTER+1))
    echo "Start run " $COUNTER
    mpirun -np 6 ./nonlinear-poro-viscoelasticity >>out.log 2>err.log
    if [ $? -eq 0 ]; then
        echo FEM Code OK
        break
    else
        echo $?
        echo FEM Code FAILED
	fi
done


# create folder for output and copy parameter file and results into it
mkdir -p $resultdir
mkdir -p $resultdir/Paraview-files

cp parameters.prm $resultdir/
cp *.sol $resultdir/
cp solution.* $resultdir/Paraview-files/

# get rid of the temporary job dir
rm -rf $tmpdir
