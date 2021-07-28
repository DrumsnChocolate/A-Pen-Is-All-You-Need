# README
developed with python 3.9\
\
installation instructions for Ubuntu, make sure you have virtualenv installed.

`python3.9 -m virtualenv venv`

`. ./venv/bin/activate`

`pip install -r requirements.txt`

And you should be good to go when it comes to the virtual environment setup.



## Work in progress, more documentation will follow soon

This repository requires two datasets from stabilo.
These are the OnHW-Dataset and a newer dataset.
The first of these can be found and downloaded at https://stabilodigital.com/data/.
After downloading this set, its contents should be placed in the root of this repository, named "IMWUT_OnHW-chars_dataset_2020-09-22".
The second of these sets is part of an ongoing competition by Stabilo, and is expected to be published soon: https://stabilodigital.com/ubicomp-2021-challenge/.
Downloading these datasets and placing them in their designated locations will make a number of symbolic links operational. These links are found in subfolders objective_1, objective_2, objective_3, objective_4 and are called data_2020 and data_2021.



