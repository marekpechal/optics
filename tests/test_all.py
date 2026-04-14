import os

for fname in os.listdir("."):
    if fname == "test_all.py":
        continue
    if fname.endswith(".py") and fname.startswith("test_"):
        os.system(f"uv run {fname}")

os.system("cd ..\\docs\\examples && uv run basic_raytracing_demo.py")
os.system("cd ..\\docs\\examples && uv run simulate_spectrometer.py")
