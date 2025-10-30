import os
import random

N_NODES = 6
N_SAMPLES = 200
OUTPUT_DIR = "generated_data"


FLOW_RANGE = (50, 200)
COMPUTE_OPTIONS = [1e8, 2e8, 3e8]
MEMORY_RANGE = (40, 120)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_matrix(active_row):
    matrix = []
    flow_value = random.randint(*FLOW_RANGE)

    for i in range(N_NODES):
        row = []
        for j in range(N_NODES):
            flow = flow_value if i == active_row else 0
            compute = random.choice(COMPUTE_OPTIONS)
            memory = random.randint(*MEMORY_RANGE)
            row.append(f"({flow},{compute:.0e},{memory})")
        matrix.append(row)
    return matrix

def save_matrix_set(sample_idx):
    filename = os.path.join(OUTPUT_DIR, f"matrix_set_{sample_idx:03d}.csv")
    with open(filename, "w", encoding="utf-8") as f:
        for active_row in range(N_NODES):
            matrix = generate_matrix(active_row)
            for row in matrix:
                f.write(" ".join(row) + "\n")
            f.write("\n")
    print(f"✅ Generated {filename}")

if __name__ == "__main__":
    for i in range(N_SAMPLES):
        save_matrix_set(i)
    print(f"\n🎯 Done! Generated {N_SAMPLES} datasets in '{OUTPUT_DIR}' folder.")
