import pylab as plt
import pystellibs
import figrc
import setup_mpl
setup_mpl.theme()
setup_mpl.solarized_colors()

lib = pystellibs.BaSeL() + pystellibs.Rauch()

for osl in lib._olist:
    l = plt.plot(osl.logT, osl.logg, 'o')[0]
    osl.plot_boundary(color=l.get_color(), dlogT=0.1, dlogg=0.3, alpha=0.3,
                      label=osl.name)

plt.xlim(5.6, 2.8)
plt.ylim(8.5, -2)
figrc.hide_axis('top right'.split())
plt.xlabel('log T$_{eff}$')
plt.ylabel('log g')
plt.tight_layout()

plt.legend(frameon=False, loc='upper left')
plt.savefig('combined_libs.png')


plt.figure()
basel = pystellibs.BaSeL()
kurucz = pystellibs.Kurucz()
ap = (4., 3.5, 0., 0.02)
sb = basel.generate_stellar_spectrum(*ap)
sk = kurucz.generate_stellar_spectrum(*ap)
# plt.loglog(osl._wavelength, osl.spectra.T, 'b-', lw=0.1, alpha=0.1, rasterized=True)
plt.loglog(osl._wavelength, sb, label='BaSel')
plt.loglog(osl._wavelength, sk, label='Kurucz')
plt.legend(frameon=False, loc='lower right')
plt.xlabel("Wavelength [{0}]".format(basel.wavelength_unit))
plt.ylabel("Flux [{0}]".format(basel.flux_units))
figrc.hide_axis('top right'.split())
plt.xlim(800, 5e4)
plt.ylim(1e25, 5e30)
plt.tight_layout()
plt.savefig('single_spec_libs.png')


# many stars
import pystellibs
from pystellibs.simpletable import SimpleTable

# A set of points (from isochrones)
t = SimpleTable('./output140512647413.dat')

# request is to have these 4 parameters: logT, logg, logL and Z
t.set_alias('logT', 'logTe')
t.set_alias('logg', 'logG')
t.set_alias('logL', 'logL/Lo')

basel = pystellibs.BaSeL()
w, f = basel.generate_individual_spectra(t)
plt.figure()
plt.loglog(w.magnitude, f.magnitude.T, 'k-', lw=0.4, alpha=0.1)
plt.xlabel("Wavelength [{0}]".format(basel.wavelength_unit))
plt.ylabel("Flux [{0}]".format(basel.flux_units))
figrc.hide_axis('top right'.split())
plt.xlim(800, 5e4)
plt.ylim(1e18, 1e34)
plt.tight_layout()
plt.savefig('many_specs_lib.png')
