<div align="center"> <img src="https://raw.githubusercontent.com/mehran-hmdpr/orcpy/main/orcpy.png" width="350" height="300" >

<div align="left">

# orcpy
**orcpy**: A lightweight Python module that applies thermodynamic principles, analysing a given set of heat source and heat sink data to find the optimum design parameters of an organic Rankine cycle for waste heat recovery projects.



## Table of contents
* [Description](#description)
     * [Problem definition](#problem-definition)
     * [Problem solution](#problem-solution)

* [Dependencies and installation](#dependencies-and-installation)

* [Examples and Tutorials](#examples-and-tutorials)

* [Authors and contributors](#authors-and-contributors)

* [License](#license)

## Description
**orcpy** is a Python package which can come to your aid to deal with *Waste heat recovery* (WHR) projects. This package can give you the optimum pressure level, working fluid and mass flow rate to obtain maximum power generation and heat recovery, considering the constraints of your project such as the minimum allowable temperature of the waste heat stream and condenser.

#### Problem definition
First step is formalization of the problem in the **orcpy** framework. The optimization problem which orcpy is desigend to solve is shown below:


*Maximization*   *f<sub>work<sub>* *(pressure level)*

   *sub. to:*
  

  
  *waste heat temperature > T<sub>dew<sub>*
  
  *temperature difference in heat exchangers* > $\Delta$*T<sub>min<sub>*
  
  *evaporator pressure* < *0.85 P<sub>critical<sub>* 
  
  *condenser temperature* > *T<sub>allowable<sub>*
  
  *condenser pressure* > *1 bar*
          
#### Problem solution
  
After defining it, we want of course to solve such a problem. To this aim, the orcpy package uses a golden section search approach to find the optimum pressure level between the upper (*0.85 P<sub>critical<sub>*) and the lower (*P<sub>condenser<sub>*) boundaries. The input variables that should be given to the orcpy are as follow:
  
- The inlet temperature for the waste stream (ºC)
- The outlet temperature for the waste stream (ºC)
- The power of the waste stream (kW)
- The heat capacity of the waste stream (kW / kgºC)
- The minimum temperature of the condenser (ºC)
- The minimum temperature difference (ºC)
- Isentropic efficiencies of turbine and pump (%)

  knowing these parameter, orcpy will give you optimum design parameters of the ORC system.
  
## Dependencies and installation
**orcpy** requires `numpy`, `pandas`, `CoolProp`, `pina`, `plotly`. The code is tested for Python 3, while compatibility of Python 2 is not guaranteed anymore. It can be installed using `pip` or directly from the source code.

### Installing via PIP
To install the package just type:
```bash
> pip install orcpy
```
To uninstall the package:
```bash
> pip uninstall orcpy
```
## Examples and Tutorials
To use orcpy after installation of requierd packages just type:
  ```bash
> from orcpy import design
  Results, figure = design.ORC.cycle("all")
```
Next, the orcpy will ask you input variables. instead of `"all"` you can also input a list of working fluids you want to analyze.  
  
## Authors and contributors
**orcpy** is developed and mantained by
* [Mehran Ahmadpour](mailto:mehran.hmdpr@gmail.com)

under the supervision of Prof. Ramin Roshandel and Prof. Mohammad B. Shafii

Contact us by email for further information or questions about **orcpy**, or suggest pull requests. Contributions improving either the code or the documentation are welcome!

## License

See the [LICENSE](https://github.com/mehran-hmdpr/orcpy/blob/main/LICENSE) file for license rights and limitations (MIT).

   
