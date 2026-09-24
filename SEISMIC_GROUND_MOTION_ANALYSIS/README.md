# SEISMIC GROUND MOTION ANALYSIS

![alt text](https://github.com/salardelavar/SALAR_PYTHON-CODE/blob/main/SEISMIC_GROUND_MOTION_ANALYSIS/SEISMIC_GROUND_MOTION_ANALYSIS.png) 

The code reads 200 seismic acceleration text files named `Ground_Acceleration_1.txt` through 
`Ground_Acceleration_200.txt`, loads each file with `np.loadtxt` (falling back to comma-delimited 
parsing if needed), selects the acceleration column if the data has multiple columns, computes the
 maximum absolute acceleration for each record, and stores the results in a NumPy array while skipping
 any missing files or NaN values. It then calculates key statistics—sample mean, standard deviation,
 minimum, quartiles Q1, Q2 (median), Q3, Q4 (maximum), and the interquartile range—and prints them to
 the console. Finally, it creates a histogram of all 200 peak values, overlays vertical reference lines
 for the mean, Q1, Q2, Q3, minimum, and Q4/maximum, adds a shaded mean ± 1σ band, inserts a text box
 summarizing the statistics, labels the axes and legend, and saves the figure as `max_acceleration_histogram.png`
 while displaying it on screen.
