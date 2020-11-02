# Bartolina

Bartolina is a real space reconstruction algorithm that corrects galaxy positions by Kaiser and Finger of God (FoG) effects. 
We follow the work carried out in [Shi et al. 2016, ApJ, 833, 241](https://iopscience.iop.org/article/10.3847/1538-4357/833/2/241/pdf).

Bartolina has the principal class:
* **ReZspace**: Sets parameters such as the cosmology to use and allows a set of methods. 

Bartolina has the following methods:
* **Halos**: Find massive dark matter haloes and cartesian coordinates of his centers. Necesary for all the other methods.
* **Kaisercorr**: Corrects the Kaiser effect only.
* **FoGcorr**: Corrects the Finger of God effect only.
* **RealSpace**: Corrects both effects (Kaiser and FoG).

## Requirements

You need Python 3.8 to run Bartolina

## Development Install

Clone this repo and then inside the local directory execute

* $ pip install -e .

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE) file for details.

## Authors

Noelia Rocío Perez and Claudio Antonio Lopez Cortez





