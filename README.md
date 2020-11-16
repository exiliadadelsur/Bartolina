# Bartolina

Bartolina is a real space reconstruction algorithm that corrects galaxy positions by Kaiser and Finger of God (FoG) effects. 
We follow the work carried out in [Shi et al. 2016, ApJ, 833, 241](https://iopscience.iop.org/article/10.3847/1538-4357/833/2/241/pdf) and [Wang et al. 2015, MNRAS, 420, 1809](https://watermark.silverchair.com/mnras0420-1809.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAArgwggK0BgkqhkiG9w0BBwagggKlMIICoQIBADCCApoGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM3j6D8TCqbxfdtDAbAgEQgIICaxRFP4w8bVEv7OgmhdIGgYzaAcPcLCBKlFqxILz4MoGLXAegkZskjI_ONw8SFFhQs9H6BymMSlwtB9m6ZvilXrIVEp5zJiXm499-RrGPyxRZa-xWg-P4AcEZONjfe-y4mLvtCv6aAp0F8AGtphDQmJv1rq3hCj94TSb5hyilWaIfmbppmd6EuGfnRon3qbU5tD1vchpQ2aFvMK9PiwLwypJZfOqJJ2pjH1t69t59bxTX0eZJl2XC2XgTeDQAkS43YFCMy0AiwlKDxx6bmqileLy-AtAlJtdMSlKu8gpw3Lit7aGWUl97vGeJ3QgNGYqVpGFJw7XfZXpqT5mHCikMoTP4C1LDNXPnlfYRIentt0QhOd2_Ktd3yH1D-PuoEwx-Cj8gsqLWKGuLAGB-k1upgm8mpUF3B6vZj56bYPxViI_X_yCjGmX_ow1MoamuNusL_dZojV6pb7jHy6XtM3RFIjDw_e8__J6JQx40XRmA1_9q48f5bMw2zFlJSem6riNSYrkjVx-ATOgAI1Hmf_BC7P2V_88TDxMnFyroYhNbIY9jEzD5LqltVqeW4xlf8omqmL7pp6xTI-Juaar0ASPMtHat06TQLAF_u1pzLLy6HKlJOUJmDu3N6gLMTih077q6l7QD-wr8G4mn4TPEfOEm4r_vR9b43fbKNSU0M3XnkRPVauxNQmLAUBmQdCt17SEnXQrASvdJ39LO105t1o-7RX7mj_JWSHxb-ZwMqEnlEJYO4GIxAr8eF3hvuZig5mplU-4uzsy4sGykd2g7U17HPk1fUkK8U3TJ2gllHxaIHf-vXITzacax3Q43Qn8).

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

```
$ pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE) file for details.

## Authors

Noelia Rocío Perez and Claudio Antonio Lopez Cortez





