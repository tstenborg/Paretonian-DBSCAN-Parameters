# Paretonian-DBSCAN-Parameters

Low-cost Pareto-like parameter estimation for sklearn's DBSCAN.

---

| Parameters                                        | Clusters | Outliers | Notes                   |
| ------------------------------------------------- | -------- | ---------| ----------------------- |
| &epsilon; = 1, minPts = 20                        | &#45;    | &#45;    | Crash, exhausts memory. |
| &epsilon; = 0.5, minPts = 5                       | 2        | 0        | Runtime ∼ minutes. Highly asymmetric clusters. |
| &epsilon; = 9 &times; 10<sup>−4</sup>, minPts = 5 | 232      | 176,195  | Runtime ∼ seconds. |

Table 1. DBSCAN clustering parameter selection scheme testing. Schemes: naive (top), sklearn default (mid), Paretonian (bottom). Test data were magnetic and gravity survey data from a site near Cloncurry (east of Mount Isa, Queensland). Adapted from Stenborg and Silversides (2022), below.

---

### Key Files

- DBSCAN_pareto.py &nbsp;&nbsp; A Python program for parameter estimation with sklearn's DBSCAN.<br />
- Data_Description.txt &nbsp;&nbsp; A description of the test data.<br />
- Mag_Grav_Grav1vd.csv &nbsp;&nbsp;&nbsp; Magnetic and gravity survey test data (source: Geoscience Australia).

### Software Requirements

- Python.<br />
- Python IDE, e.g., Visual Studio Code (optional).<br />

### Reference

Stenborg, TN, Silversides, K 2022, "[Low-cost Paretonian DBSCAN Parameter Estimation for Sklearn](https://www.australiandatascience.net/wp-content/uploads/2022/11/ADSN22_Proceedings.pdf)" Proc. Australian Data Science Network Conf. 2022, Australian Data Science Network, pp. 8&ndash;9.
