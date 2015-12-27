T-61.5910: Research Project in Computer and Information Science

A Comparison of Univariate and Multivariate Methods to Forecast the Economy

Report by:
Rodrigues Pereira, Ronnie – 522711

Advisor:
Pyry Takala

___________________________
Python Script

1) Retain the folder structure, because the script uses relative paths;

2) The datasets must be analyzed separately (There are two flags - sd, fi);

3) Usage:
		full_model.py -flag
			-sd: Forecast Simulated Data (Default)
			-fi: Forecast Financial Data

N.B.: The script catches many exceptions, due to the parameter selection. Therefore, the algorithms run silently whenever possible. Furthermore, exceptions that were issued to stderr are captured. However, the user might still face warning messages from other dependencies.

Minimum Requirements:
		- Python 2.7
		- Pandas 0.16.2
		- Numpy v.1.9.2
		- StatsModels v.0.60
		- SciKit Learn v.0.17