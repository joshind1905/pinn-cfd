import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _pdist(x):
   
	x = np.atleast_2d(x)
	assert len(x.shape)==2, 'Input array must be 2d-dimensional'
    
	m, n = x.shape
	if m<2:
		return []
    
	d = []
	for i in range(m - 1):
		for j in range(i + 1, m):
			d.append((sum((x[j, :] - x[i, :])**2))**0.5)
    
	return np.array(d)




def lhsclassic(n, samples):
    # Generate the intervals
	cut = np.linspace(0, 1, samples + 1)    

	# Fill points uniformly in each interval
	u = np.random.rand(samples, n)
	a = cut[:samples]
	b = cut[1:samples + 1]
	rdpoints = np.zeros_like(u)
	for j in range(n):
		rdpoints[:, j] = u[:, j]*(b-a) + a
	    
	# Make the random pairings
	H = np.zeros_like(rdpoints)
	for j in range(n):
		order = np.random.permutation(range(samples))
		H[:, j] = rdpoints[order, j]

	return H



def lhsmaximin(n, samples, iterations):
	maxdist = 0
    
	# Maximize the minimum distance between points
	for i in range(iterations):
		Hcandidate = lhsclassic(n, samples)
        
		d = _pdist(Hcandidate)
		#print(d)
		if maxdist<np.min(d):
			#print(np.min(d))
			maxdist = np.min(d)
			H = Hcandidate.copy()
    
	return H



def sequential_lhsmaximin(n, samples, iterations, previousLHS):
	maxdist = 0
    
	# Maximize the minimum distance between points
	for i in range(iterations):
		Hcandidate = lhsclassic(n, samples)

		Hcomplete = np.concatenate((previousLHS, Hcandidate))
		#print(Hcomplete)

		d = _pdist(Hcomplete)
		if maxdist<np.min(d):
			maxdist = np.min(d)
			H = Hcandidate.copy()
    
	return H

# Function to rescale the DOE values to actual ranges
def rescale(doe, ranges):
    scaled_doe = np.zeros_like(doe)
    for i in range(doe.shape[1]):
        min_val, max_val = ranges[i]
        # If we're scaling the COC variable (6th variable, index 5)
        if i == 5:
            # Apply logarithmic scaling
            scaled_doe[:, i] = 10**(doe[:, i] * (np.log10(max_val) - np.log10(min_val)) + np.log10(min_val))
        else:
            # Apply linear scaling for all other variables
            scaled_doe[:, i] = doe[:, i] * (max_val - min_val) + min_val
    return scaled_doe
    
###########################################################################   
###########################################################################
# Define the ranges for each parameter
ranges = [
    [0.1, 1],  # U_ave
    [0.001, 0.1],  # kin_vis
]
###########################################################################
###########################################################################

n_parameters = 2
n_samples = 20

step = 1

if step == 1:
	doe = lhsmaximin(n_parameters, samples=n_samples, iterations=300)

	csv_filename = 'doe.csv'

	with open(csv_filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		# Write a header row, if desired
		writer.writerow(['U_ave', 'kin_vis'])
		# Write the data rows
		writer.writerows(doe)

	print(f"{csv_filename} has been created with the fractional factorial design.")

if step > 1:
	more_samples = 30
	previousLHS = np.loadtxt("doe.csv",delimiter=",", skiprows=1)
	sequencial_doe = sequential_lhsmaximin(n_parameters, samples=more_samples, iterations=1000, previousLHS = previousLHS)

	with open('Originaldoe.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		# Write a header row, if desired
		writer.writerow(['U_ave', 'kin_vis'])
		# Write the data rows
		writer.writerows(previousLHS)

	csv_filename = 'doe.csv'
	with open(csv_filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		# Write a header row, if desired
		writer.writerow(['U_ave', 'kin_vis'])
		# Write the data rows
		writer.writerows(previousLHS)
		writer.writerows(sequencial_doe)

	print(f"{csv_filename} has been created with the fractional factorial design.")



### rescaling ####
# Rescale the DOE
doe = np.loadtxt("doe.csv",delimiter=",", skiprows=1)
rescaled_doe = rescale(doe, ranges)
#rescaled_doe = np.round(rescaled_doe, 2)

csv_filename = 'doe_scaled.csv'

rescaled_doe[:,0] = np.round(rescaled_doe[:,0], 2)
rescaled_doe[:,1] = np.round(rescaled_doe[:,1], 6)
#rescaled_doe[:,2] = np.round(rescaled_doe[:,2], 2)
#rescaled_doe[:,3] = np.round(rescaled_doe[:,3], 2)
#rescaled_doe[:,4] = np.round(rescaled_doe[:,4], 2)
#rescaled_doe[:,5] = np.round(rescaled_doe[:,5], 2)


# Write the rescaled DOE to a CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write a header row, if desired
    writer.writerow(['U_ave', 'kin_vis', 'H', 'L', 'h', 'l'])
    # Write the data rows with rescaled values
    for row in rescaled_doe:
        writer.writerow(row)

print(f"{csv_filename} has been created with the rescaled fractional factorial design.")


# Convert the rescaled DOE array into a Pandas DataFrame
column_names = ['U_ave', 'kin_vis']
df_rescaled = pd.DataFrame(rescaled_doe, columns=column_names)

# Create a pairplot with seaborn
g = sns.pairplot(df_rescaled, diag_kind='kde')




# Adjust the plot size and layout as necessary
g.fig.set_size_inches(15, 15)
plt.tight_layout()
#plt.show()
plt.savefig('doe_pairplot.png', dpi=300)  # Save the figure with 300 dpi
