from pystellibs import libraries


def test_generate_a_single_spectrum():
    """Generate a single spectrum from two different libraries and plot them

    Test only the grids available in the repo (primarily small ones).
    The others are too large to be included in the repo.

    this test makes sure that the libraries can be loaded and a spectrum generated when not extrapolating.
    """
    # create atmosphere libraries
    ap = (4.0, 3.5, 0.0, 0.02)
    for libname, lib in libraries.items():
        print("Testing library:", libname)
        try:
            sl = lib()
            _ = sl.generate_stellar_spectrum(*ap, raise_extrapolation=False)
            print(f"   - library {sl.name} - ok")
        except FileNotFoundError:
            print(f"   - Library {libname} not found. Skipping...")
            continue


def test_extrapolation_warning():
    """Test that extrapolation warning is raised when extrapolating"""
    ap = (4.0, 3.5, 0.0, 0.02)
    for libname in ["rauch"]:
        lib = libraries[libname]
        print("Testing library:", libname)
        try:
            sl = lib()
        except FileNotFoundError:
            print(f"   - Library {libname} source file not found. Skipping...")
            continue
        try:
            print("   - Testing extrapolation error...")
            _ = sl.generate_stellar_spectrum(*ap, raise_extrapolation=True)
        except RuntimeError:
            print("   - Extrapolation error raised.")
        except Exception as e:
            raise AssertionError(f"Unexpected exception {e} for library {libname}")
