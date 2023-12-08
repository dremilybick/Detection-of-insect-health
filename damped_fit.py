from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

field = 'corn1'

snippets = pd.read_csv(f'results/data/{field}/events.csv')
# From this I just want the length and the intensity per snippet, so you can use your summarised ones
snippets = snippets[snippets['length']>0]

# Equation for damped oscillator
def damped_oscillator(x, m, c, p):
    return (-2*m/c)*(np.log(0.0001)-np.log(x)) - p

# Initial curve fit
params, _ = curve_fit(damped_oscillator, snippets['intensity'], snippets['length'])
m, c, p = params

# Find any outliers (generally caused by double events)
y_fit = [damped_oscillator(x, m, c, p) for x in snippets['intensity']]
residuals = snippets['length'] - y_fit
threshold = 2 * np.std(residuals)
outliers = np.abs(residuals) > threshold

# Fit again minus the outliers
params, _ = curve_fit(damped_oscillator, snippets[~outliers]['intensity'], snippets[~outliers]['length'])
m, c, p = params

# Pearson R of fit
yfit = [damped_oscillator(x, m, c, p) for x in snippets['intensity']]
r, pval = pearsonr(snippets['length'], yfit)
if pval < 0.0001:
    pval = '< 0.0001'
else:
    pval = round(pval,4)
rsquare = r**2

# Plot intensity/length of data with trend line
x_vals = np.arange(min(snippets['intensity']),max(snippets['intensity']),max(snippets['intensity'])/100)
y = [damped_oscillator(x, m, c, p) for x in x_vals]
plt.plot(x_vals, y)
plt.scatter(snippets['intensity'], snippets['length'])
plt.xlabel('max intensity')
plt.ylabel('length')
plt.title(f"{field}, r2={round(rsquare, 3)}, pval={pval}")
plt.tight_layout()
plt.show()