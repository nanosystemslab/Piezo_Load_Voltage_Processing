import matplotlib.pyplot as plt

# Data
force = [0, 100, 200, 300, 400, 500]  # Force (N)
electric_displacement = [0, 2.47E-10, 4.95E-10, 7.42E-10, 9.89E-10, 1.24E-09]  # Electric displacement field norm (C)

# Figure dimensions
figsize = 4  # Custom figure size
figdpi = 600  # High DPI for better resolution
hwratio = 4./3  # Height-to-width ratio

# Create figure with custom dimensions
fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
ax = fig.add_subplot(111)

# Plot
ax.plot(force, electric_displacement, marker='o', linestyle='-', color='b', linewidth=3, markersize=10, label="Electric Displacement vs. Force")

# Labels and title with larger font sizes
ax.set_xlabel("Force (N)", fontsize=13, fontweight='bold')
ax.set_ylabel("Electric Displacement Field Norm (C)", fontsize=12, fontweight='bold')
#ax.set_title("Force vs. Electric Displacement Field Norm", fontsize=18, fontweight='bold')

# Legend with larger font
#ax.legend(fontsize=14)

# Turn off the grid
ax.grid(False)

# Increase tick label size
ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()  # Automatically adjusts layout
#plt.subplots_adjust(bottom=0.25) 

# Saving plot 
outputfile = "out/ChargevsForce.png"
plt.savefig(outputfile, dpi=600)
# Show plot
plt.show()
