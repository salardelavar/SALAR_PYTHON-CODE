"""
============================================================
COMPREHENSIVE FIELD SURVEY DATA PROCESSING SYSTEM
============================================================
Author: SALAR DELAVAR GHASHGHAEI (QASHQAI)
Version: 1.0

DESCRIPTION:
This script processes field survey data collected from a central
observation point using laser distance meter and magnetic compass.
It performs:
1. Polar to Cartesian coordinate transformation
2. Visualization of point distribution
3. Polygon area and perimeter calculations
4. Geometric property analysis
5. Convex hull computations

INPUT:
- Distances from observer to points (meters)
- Magnetic bearings from North (degrees, clockwise)

OUTPUT:
- Cartesian coordinates of all points
- Visualizations (Cartesian and Polar plots)
- Geometric properties (area, perimeter, angles)
- Summary statistics and metrics
============================================================

This code is a comprehensive field survey data processing system that,
 given the distance and angle of each point relative to a central point
 (measured with a laser meter and compass), calculates the points’ Cartesian coordinates
 visualizes their positions, computes the area and perimeter of the polygon formed by the points,
 analyzes geometric properties such as angles, centroid, and distances, and finally produces 
 Cartesian and polar plots along with a summary table of all relevant information, allowing 
 an ordinary person to clearly see and understand the shape and dimensions of the surrounding points.

"""
import numpy as np
import matplotlib.pyplot as plt
#import math
#%% --------------------------------------------------
# Input data: distances from you to each point (meters) and angles relative to Magnetic North
lengths = [4.354, 4.238, 6.204, 5.800, 6.054, 4.028, 2.354]
angles_deg = [165, 184, 255, 261, 267, 300, 91]
#%% --------------------------------------------------
# You are at the coordinate center (0,0)
center_x = 0.0
center_y = 0.0

# Convert angles: Magnetic North (clockwise) → Standard mathematical angle (counterclockwise from x+ axis)
# In mathematics: North = 90°, East = 0°
# Formula: math_angle = 90 - magnetic_angle
angles_math_deg = []
for alpha in angles_deg:
    theta = 90 - alpha
    angles_math_deg.append(theta)

# Convert to radians
angles_rad = [np.radians(theta) for theta in angles_math_deg]
#%% --------------------------------------------------
# Calculate Cartesian coordinates for each point
points_x = []
points_y = []
for i in range(len(lengths)):
    # In standard mathematics: x = r * cos(θ), y = r * sin(θ)
    x = lengths[i] * np.cos(angles_rad[i])
    y = lengths[i] * np.sin(angles_rad[i])
    points_x.append(x)
    points_y.append(y)

# Display results
print("=" * 60)
print("You are at the central point (0, 0).")
print("Coordinates of surrounding points (meters):")
print("=" * 60)
for i in range(len(points_x)):
    print(f"Point {i+1}: Distance = {lengths[i]:.3f} m, "
          f"Magnetic angle = {angles_deg[i]}°, "
          f"Coordinates = ({points_x[i]:.3f}, {points_y[i]:.3f})")
print("=" * 60)
#%% --------------------------------------------------
# Create visualization
plt.figure(figsize=(12, 10))

# Plot measured points
plt.scatter(points_x, points_y, c='red', s=200, zorder=5, label='Measured Points')
for i, (x, y) in enumerate(zip(points_x, points_y)):
    plt.text(x, y + 0.2, f'P{i+1}\n({angles_deg[i]}°)', 
             ha='center', va='bottom', fontsize=9, weight='bold')

# Your position at the center
plt.scatter([center_x], [center_y], c='blue', s=300, marker='*', 
            label='Your Position (Center)', zorder=10)
plt.text(0, 0.3, 'YOU', ha='center', va='bottom', fontsize=12, weight='bold')

# Lines from you to each point
for i in range(len(points_x)):
    plt.plot([center_x, points_x[i]], [center_y, points_y[i]], 
             'gray', linestyle='--', alpha=0.7, linewidth=1)

# Distance circles (for scale)
for r in [2, 4, 6, 8]:
    circle = plt.Circle((0, 0), r, color='lightgray', fill=False, linestyle=':', alpha=0.5)
    plt.gca().add_artist(circle)
    plt.text(r, 0.2, f'{r}m', ha='left', va='bottom', fontsize=8, color='gray')

# Cardinal directions
arrow_length = max(lengths) * 1.2
# North (90° in mathematics)
plt.arrow(0, 0, 0, arrow_length/2, head_width=0.3, head_length=0.5, 
          fc='green', ec='green', alpha=0.8, label='Magnetic North')
plt.text(0.2, arrow_length/2, 'NORTH', color='green', weight='bold')

# East (0° in mathematics)
plt.arrow(0, 0, arrow_length/2, 0, head_width=0.3, head_length=0.5, 
          fc='orange', ec='orange', alpha=0.8, label='East')
plt.text(arrow_length/2, 0.2, 'EAST', color='orange', weight='bold')

# South (270° in mathematics)
plt.arrow(0, 0, 0, -arrow_length/2, head_width=0.3, head_length=0.5, 
          fc='green', ec='green', alpha=0.5, linestyle=':')
plt.text(0.2, -arrow_length/2, 'SOUTH', color='green', alpha=0.7)

# West (180° in mathematics)
plt.arrow(0, 0, -arrow_length/2, 0, head_width=0.3, head_length=0.5, 
          fc='orange', ec='orange', alpha=0.5, linestyle=':')
plt.text(-arrow_length/2, 0.2, 'WEST', color='orange', alpha=0.7)

# Chart settings
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('East (meters)')
plt.ylabel('North (meters)')
plt.title('Position of Points Around You (Measured with Laser Meter and Compass)\n' +
          'Angles relative to Magnetic North, clockwise direction')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
#%% --------------------------------------------------
# Set display range
max_range = max(lengths) * 1.3
plt.xlim(-max_range, max_range)
plt.ylim(-max_range, max_range)

# Display coordinate table beside the chart
plt.figtext(1.02, 0.95, "Coordinate Table:", fontsize=11, weight='bold')
table_text = ""
for i in range(len(points_x)):
    table_text += f"P{i+1}: ({points_x[i]:.2f}, {points_y[i]:.2f})\n"
plt.figtext(1.02, 0.85, table_text, fontfamily='monospace', fontsize=10)

plt.show()
#%% --------------------------------------------------
# Additional information
print("\nSummary Information:")
print("-" * 40)
print(f"Closest point: P{np.argmin(lengths)+1} ({min(lengths):.2f} meters)")
print(f"Farthest point: P{np.argmax(lengths)+1} ({max(lengths):.2f} meters)")

# Calculate angles between points (optional)
if len(points_x) >= 2:
    print("\nAngles between points from your perspective:")
    for i in range(len(points_x)):
        for j in range(i+1, len(points_x)):
            # Dot product to calculate angle
            dot = points_x[i]*points_x[j] + points_y[i]*points_y[j]
            norm_i = np.sqrt(points_x[i]**2 + points_y[i]**2)
            norm_j = np.sqrt(points_x[j]**2 + points_y[j]**2)
            if norm_i * norm_j > 0:
                cos_angle = dot / (norm_i * norm_j)
                cos_angle = max(-1, min(1, cos_angle))  # Limit to range [-1,1]
                angle_between = np.degrees(np.arccos(cos_angle))
                print(f"Angle P{i+1}─You─P{j+1}: {angle_between:.1f}°")
#%% --------------------------------------------------
# Calculate and display bearing from you to each point
print("\n" + "=" * 60)
print("Bearing from your position to each point:")
print("=" * 60)
for i in range(len(points_x)):
    dx = points_x[i] - center_x
    dy = points_y[i] - center_y
    
    # Calculate mathematical angle
    math_angle_rad = np.arctan2(dy, dx)
    math_angle_deg = np.degrees(math_angle_rad)
    
    # Convert to magnetic bearing (clockwise from North)
    magnetic_bearing = (90 - math_angle_deg) % 360
    
    print(f"P{i+1}: Bearing = {magnetic_bearing:.1f}°, "
          f"Distance = {lengths[i]:.2f} m, "
          f"Direction = ", end="")
    
    # Add compass direction description
    if magnetic_bearing >= 337.5 or magnetic_bearing < 22.5:
        print("N")
    elif magnetic_bearing < 67.5:
        print("NE")
    elif magnetic_bearing < 112.5:
        print("E")
    elif magnetic_bearing < 157.5:
        print("SE")
    elif magnetic_bearing < 202.5:
        print("S")
    elif magnetic_bearing < 247.5:
        print("SW")
    elif magnetic_bearing < 292.5:
        print("W")
    else:
        print("NW")
#%% --------------------------------------------------
# Create a polar plot for better angular visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Cartesian plot (left)
ax1.scatter(points_x, points_y, c='red', s=100)
ax1.scatter([0], [0], c='blue', s=200, marker='*')
for i in range(len(points_x)):
    ax1.plot([0, points_x[i]], [0, points_y[i]], 'gray', linestyle='--', alpha=0.7)
ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.set_title('Cartesian Coordinates')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')
for i, (x, y) in enumerate(zip(points_x, points_y)):
    ax1.text(x, y, f' P{i+1}', va='bottom')

# Polar plot (right)
ax2 = plt.subplot(122, projection='polar')
# Convert magnetic angles to radians for polar plot
angles_rad_magnetic = [np.radians(alpha) for alpha in angles_deg]
ax2.scatter(angles_rad_magnetic, lengths, c='red', s=100)
ax2.set_theta_zero_location('N')  # North at top
ax2.set_theta_direction(-1)  # Clockwise direction (magnetic compass)
ax2.set_title('Polar Plot (Angles from Magnetic North, clockwise)')
ax2.grid(True)
for i in range(len(lengths)):
    ax2.text(angles_rad_magnetic[i], lengths[i], f' P{i+1}', va='bottom')

plt.tight_layout()
plt.show()

#-----------------------------------------------
# Convert angles to Cartesian coordinates
angles_math_deg = [90 - alpha for alpha in angles_deg]
angles_rad = [np.radians(theta) for theta in angles_math_deg]

# Calculate Cartesian coordinates
points = []
for i in range(len(lengths)):
    x = lengths[i] * np.cos(angles_rad[i])
    y = lengths[i] * np.sin(angles_rad[i])
    points.append([x, y])

# Convert to numpy array
points = np.array(points)

# FUNCTION 1: Calculate polygon perimeter (connecting points in order)
def calculate_perimeter(points):
    perimeter = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        distance = np.linalg.norm(points[j] - points[i])
        perimeter += distance
    return perimeter

# FUNCTION 2: Calculate polygon area using Shoelace formula
def calculate_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area
#%% --------------------------------------------------
# Calculate properties
print("=" * 70)
print("POLYGON AREA AND PERIMETER CALCULATIONS")
print("=" * 70)

# 1. Polygon formed by connecting points in given order
print("\n1. POLYGON (points connected in given order):")
print("-" * 40)
perimeter = calculate_perimeter(points)
area = calculate_area(points)
print(f"Perimeter: {perimeter:.3f} meters")
print(f"Area: {area:.3f} square meters")

# Calculate distances between consecutive points
print("\nSegment lengths:")
for i in range(len(points)):
    j = (i + 1) % len(points)
    distance = np.linalg.norm(points[j] - points[i])
    print(f"  P{i+1} to P{j+1}: {distance:.3f} m")

#%% --------------------------------------------------
# 2. Additional geometric properties
print("\n3. ADDITIONAL GEOMETRIC PROPERTIES:")
print("-" * 40)

# Calculate distances from centroid
centroid = np.mean(points, axis=0)
print(f"Centroid coordinates: ({centroid[0]:.3f}, {centroid[1]:.3f})")

# Calculate distances from centroid to vertices
print("\nDistances from centroid to vertices:")
max_distance = 0
min_distance = float('inf')
for i, point in enumerate(points):
    distance = np.linalg.norm(point - centroid)
    print(f"  P{i+1}: {distance:.3f} m")
    max_distance = max(max_distance, distance)
    min_distance = min(min_distance, distance)

print(f"\nMaximum distance from centroid: {max_distance:.3f} m")
print(f"Minimum distance from centroid: {min_distance:.3f} m")
print(f"Area/Perimeter ratio: {area/perimeter:.4f} (m²/m)")
#%% --------------------------------------------------
# 3. Calculate interior angles
print("\n4. INTERIOR ANGLES:")
print("-" * 40)
for i in range(len(points)):
    # Get three consecutive points
    prev = points[(i - 1) % len(points)]
    curr = points[i]
    next_pt = points[(i + 1) % len(points)]
    
    # Vectors from current point
    v1 = prev - curr
    v2 = next_pt - curr
    
    # Calculate angle between vectors
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 * norm_v2 > 0:
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        print(f"  Angle at P{i+1}: {angle_deg:.1f}°")
    
    # Check if polygon is convex
    cross_product = np.cross(v1, v2)
    if i == 0:
        sign = np.sign(cross_product)
        is_convex = True
    elif sign * cross_product < 0:
        is_convex = False

print(f"\nPolygon is convex: {is_convex}")
#%% --------------------------------------------------
# VISUALIZATION
fig, (ax1) = plt.subplots(1, figsize=(14, 6))

# Plot 1: Polygon with measurements
ax1.set_aspect('equal')
ax1.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.7)
ax1.plot(np.append(points[:, 0], points[0, 0]), 
         np.append(points[:, 1], points[0, 1]), 'b-', linewidth=2, alpha=0.7)
ax1.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5)

# Plot centroid
ax1.scatter(centroid[0], centroid[1], c='green', s=200, marker='*', label='Centroid')

# Label points
for i, (x, y) in enumerate(points):
    ax1.text(x, y, f' P{i+1}', fontsize=10, ha='left', va='bottom')

# Add area and perimeter text
ax1.text(0.02, 0.98, f'Perimeter: {perimeter:.2f} m\nArea: {area:.2f} m²', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.set_title('Polygon Formed by Connecting Points')
ax1.grid(True, alpha=0.3)
ax1.legend()
plt.show()
#%% --------------------------------------------------
# 4. Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Property':<25} {'Value':<20} {'Units':<10}")
print("-" * 70)
print(f"{'Polygon Perimeter':<25} {perimeter:<20.3f} {'meters':<10}")
print(f"{'Polygon Area':<25} {area:<20.3f} {'m²':<10}")
print(f"{'Number of Sides':<25} {len(points):<20} {'':<10}")
print(f"{'Centroid X':<25} {centroid[0]:<20.3f} {'meters':<10}")
print(f"{'Centroid Y':<25} {centroid[1]:<20.3f} {'meters':<10}")
print(f"{'Area/Perimeter Ratio':<25} {area/perimeter:<20.4f} {'m²/m':<10}")
print(f"{'Maximum Span':<25} {2*max_distance:<20.3f} {'meters':<10}")
print("=" * 70)

# Calculate approximate equivalent circle properties
print("\n6. EQUIVALENT CIRCLE PROPERTIES (for comparison):")
print("-" * 40)
equivalent_radius = np.sqrt(area / np.pi)
equivalent_circumference = 2 * np.pi * equivalent_radius
print(f"If the area were a circle:")
print(f"  Radius: {equivalent_radius:.3f} meters")
print(f"  Diameter: {2*equivalent_radius:.3f} meters")
print(f"  Circumference: {equivalent_circumference:.3f} meters")
print(f"  Polygon perimeter is {perimeter/equivalent_circumference:.1%} of equivalent circle circumference")
#%% --------------------------------------------------
# Calculate compactness ratio
compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
print(f"\nShape compactness (4πA/P²): {compactness:.3f}")
print("(1.0 = perfect circle, <1.0 = less compact)")
#%% --------------------------------------------------
