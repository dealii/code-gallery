Nonlinear heat transfer with TRILINOS
======================================
# Introduction
This application uses automatic differentiation to develop the residual and the jacobian for a simple transient non-linear heat transfer problem. The nonlinear solution is then done using TRILINOS NOX. The nonlinearity arises due to the dependence of thermal conductivity on the temperature. This example can be found in the following [link](https://www.mathworks.com/help/pde/ug/heat-transfer-problem-with-temperature-dependent-properties.html). 
# About the folders
The source codes are there in the [source](./source) folder. Note that the geometry and its mesh are present in [mesh](./mesh) folder. The [output](./output) folder will contain the vtu files from the run. The folder [include](./include) contains all the header files needed. Further, the file [CMakeLists.txt](./CMakeLists.txt) is added so that you can run the programme as  

	cmake .
  	make all
  	make run
# Documentation
The [Documentation](./Documentation) folder has a pdf called "Non-linear_heat_conduction.pdf" which has a detailed derivation and explanation of important functions. 
# The dealii steps
To understand this application, step-71, step-72 and step-77 are needed. 
