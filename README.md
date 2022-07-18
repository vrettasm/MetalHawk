# MetalHawk

![Logo](./logos/metalhawk_logo.jpg)

This repository provides a Python implementation of the MetalHawk program
to predict metal sites classes.

M. Vrettas, PhD.

## Installation

There are two options to install the software.

1. The easiest way is to visit the GitHub web-page of the project and
[download the code](https://github.com/vrettasm/MetalHawk/archive/master.zip)
in zip format. This option does not require a prior installation of git on the
computer.

2. Alternatively one can clone the project directly using git as follows:

    `$ git clone https://github.com/vrettasm/MetalHawk.git`

## Required packages

The recommended version is **Python 3.8** (and above). Some required packages
are:

> scipy, numpy, pathlib, pandas, etc.

To simplify the required packages just use:

    $ pip install -r requirements.txt

## Virtual environment (recommended)

It is highly advised to create a separate virtual environment to avoid
messing with the main Python installation. On Linux and macOS systems
type:

    $ python3 -m venv metalhawk_venv

Note: "metalhawk_venv" is an _optional_ name.

Once the virtual environment is created activate it with:

    $ source metalhawk_venv/bin/activate

Then we can install all the requirements as above:

    $ pip3 install -r requirements.txt

or

    $ python3 -m pip install -r requirements.txt

N.B. For Windows systems follow the **equivalent** instructions.

## How to run

To execute the program (within the activated virtual environment), you can either
navigate  to the main directory of the project (i.e. where the metalhawk.py is located),
or locate it through the command line and then run the following command:

    $ ./metalhawk.py -f path/to/filename.pdb

This is the simplest way to run MetalHawk. 

   > **Hint**: If you want to run the program on multiple files (in the same directory)
   > you can use the '*' wildcard as follows:
   >  
   > $ ./metalhawk.py -f path/to/*.pdb

This will run MetalHawk on all the files (in the directory) with the '.pdb' extension.

Additionally, if you want the output to be saved in a csv file format, use the
'-o path/to/save/' option. This will use the path "path/to/save/" and create
a file with the output of the prediction. The filename is generated automatically.

---

To explore all the options of NapShift, use:

    $ ./metalhawk.py -h

You will see the following menu:

![Help](./logos/help_menu.png)

## References (and documentation)

The original work is described in detail at:

Gianmattia Sguelia, Michail D. Vrettas, Marco Chino, Angela Lombardi
and Alfonso De Simone (2022). _"TBD"_ Submitted for publication.

### Contact

For any questions/comments (**_regarding this code_**) please contact me at:
vrettasm@gmail.com