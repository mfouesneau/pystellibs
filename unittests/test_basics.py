import pylab as plt
from pystellibs import libraries

def test_generate_a_single_spectrum():
    """Generate a single spectrum from two different libraries and plot them"""
    # create atmosphere libraries
    ap = (4., 3.5, 0., 0.02)
    for lib in libraries:
        print("Testing library:", lib)
        try:
            sl = lib()
        except FileNotFoundError:
            print(f"Library {lib} not found. Skipping...")
            continue
        _ = sl.generate_stellar_spectrum(*ap)
        
        