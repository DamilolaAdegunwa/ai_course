import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(6, 6))
gs = GridSpec(3, 3, figure=fig)  # Define a 3x3 grid

# Create subplots in specific grid positions
ax1 = fig.add_subplot(gs[0, :])  # First row, spanning all columns
ax2 = fig.add_subplot(gs[1:, 0])  # First column, spanning two rows
ax3 = fig.add_subplot(gs[1, 1:])  # Second row, spanning two columns
ax4 = fig.add_subplot(gs[2, 1])  # Single cell
ax5 = fig.add_subplot(gs[2, 2])  # Single cell

# Set titles for clarity
ax1.set_title("Row 1, all columns")
ax2.set_title("Col 1, rows 2-3")
ax3.set_title("Row 2, cols 2-3")
ax4.set_title("Row 3, col 2")
ax5.set_title("Row 3, col 3")

plt.tight_layout()
plt.show()
