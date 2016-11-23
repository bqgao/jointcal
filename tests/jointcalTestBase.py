# See COPYRIGHT file at the top of the source tree.

from __future__ import division, absolute_import, print_function

import numpy as np
import collections
import os

# NOTE: We use both astropy.units and afw.arcseconds in places because:
# 1. astropy.units behaves well with matplotlib, while afw.arcseconds does not.
# 2. afw objects (e.g. Coord) need afw.arcseconds, not astropy.units.
from astropy import units as u

from lsst.meas.astrom import LoadAstrometryNetObjectsTask, LoadAstrometryNetObjectsConfig
import lsst.afw.geom
import lsst.afw.table

from lsst.jointcal import jointcal


class JointcalTestBase(object):
    """
    Base class for jointcal tests, to genericize some test running and setup.

    Derive from this first, then from TestCase.
    """

    def setUp(self):
        self.do_plot = False  # don't make plots unless specifically requested
        self.match_radius = 0.1*lsst.afw.geom.arcseconds  # match sources within 0.1" for RMS statistics
        self.all_visits = None  # list of the available visits to generate the parseAndRun arguments
        self.other_args = []  # optional other arguments for the butler dataId
        # Signal/Noise (flux/fluxSigma) for sources to be included in the RMS cross-match.
        # 100 is a balance between good centroids and enough sources.
        self.flux_limit = 100

    def tearDown(self):
        if getattr(self, 'reference', None) is not None:
            del self.reference
        if getattr(self, 'oldWcsList', None) is not None:
            del self.oldWcsList
        if getattr(self, 'jointcalTask', None) is not None:
            del self.jointcalTask

    def _prep_reference_loader(self, center, radius):
        """
        !Setup an astrometry.net reference loader.

        @param center (afw.coord) The center of the field you're testing on.
        @param radius (afw.geom.angle) The radius to load objects around center.
        """
        refLoader = LoadAstrometryNetObjectsTask(LoadAstrometryNetObjectsConfig())
        # Make a copy of the reference catalog for in-memory contiguity.
        self.reference = refLoader.loadSkyCircle(center, radius, filterName='r').refCat.copy()

    def _testJointCalTask(self, nCatalogs, relative_error, absolute_error):
        """Test parseAndRun for jointcal on nCatalogs, requiring less than some error (arcsec)."""

        self.visit_list = self.all_visits[:nCatalogs]
        visits = '^'.join(str(v) for v in self.visit_list)
        output_dir = os.path.join('.test', self.__class__.__name__)
        args = [self.input_dir, '--output', output_dir,
                '--clobber-versions', '--clobber-config',
                '--doraise',
                '--id', 'visit=%s'%visits]
        args.extend(self.other_args)
        result = jointcal.JointcalTask.parseAndRun(args=args, doReturnResults=True)
        self.assertNotEqual(result.resultList, [], 'resultList should not be empty')
        self.dataRefs = result.resultList[0].result.dataRefs
        self.oldWcsList = result.resultList[0].result.oldWcsList

        rms_rel, rms_abs = self.compute_rms(self.dataRefs)
        self.assertLess(rms_rel, relative_error)
        self.assertLess(rms_abs, absolute_error)

    def compute_rms(self, dataRefs):
        """
        !Match all dataRefs to compute the RMS, for all detections above self.flux_limit.

        @param dataRefs (list) the dataRefs to do the relative matching between

        @return (geom.Angle, geom.Angle) relative and absolute RMS of stars
        """
        visits_per_dataRef = [dataRef.dataId['visit'] for dataRef in dataRefs]

        def compute(catalogs):
            """Compute the relative and absolute RMS."""
            visitCatalogs = self._make_visit_catalogs(catalogs, visits_per_dataRef)
            refCat = visitCatalogs.values()[0]  # use the first catalog as the relative reference catalog
            relative = self._make_match_dict(refCat, visitCatalogs.values()[1:])
            absolute = self._make_match_dict(self.reference, visitCatalogs.values())
            return visitCatalogs, relative, absolute

        old_cats = [dataRef.get('src') for dataRef in dataRefs]
        visitCatalogs, old_relative, old_absolute = compute(old_cats)

        # Update coordinates with the new wcs including distortion corrections.
        new_cats = []
        for dataRef in dataRefs:
            new_cats.append(dataRef.get('src'))
            lsst.afw.table.utils.updateSourceCoords(dataRef.get('wcs').getWcs(), new_cats[-1])
        visistCatalogs, new_relative, new_absolute = compute(new_cats)

        old_rel_total = rms_total(old_relative)
        new_rel_total = rms_total(new_relative)
        old_abs_total = rms_total(old_absolute)
        new_abs_total = rms_total(new_absolute)

        if self.do_plot:
            self._do_plots(dataRefs, visitCatalogs,
                           old_relative, old_absolute, new_relative, new_absolute,
                           old_rel_total, old_abs_total, new_rel_total, new_abs_total)

        return new_rel_total, new_abs_total

    def _make_match_dict(self, reference, visitCatalogs):
        """
        !Return a dict of starID:[distances] over the catalogs, for RMS calculations.

        @param reference (SourceCatalog) catalog to do the matching against
        @param visits (list) visit source catalogs (from _make_visit_catalogs) to cross-match

        @return (dict) dict of starID:list(measurement deltas for that star)
        """

        deltas = collections.defaultdict(list)
        for cat in visitCatalogs:
            good = (cat.get('base_PsfFlux_flux')/cat.get('base_PsfFlux_fluxSigma')) > self.flux_limit
            # things the classifier called stars are not extended.
            good &= (cat.get('base_ClassificationExtendedness_value') == 0)
            matches = lsst.afw.table.matchRaDec(reference, cat[good], self.match_radius)
            for m in matches:
                deltas[m[0].getId()].append(m[2])
        return deltas

    def _make_visit_catalogs(self, catalogs, visits):
        """
        !Return a dict of visit: catalog of all sources from all CCDs of that visit.

        @param catalogs (list of SourceCatalogs) catalogs to combine into per-visit catalogs.
        @param visits (list) list of visit identifiers, one-to-one correspondant with catalogs.
        """
        visit_dict = {v: lsst.afw.table.SourceCatalog(catalogs[0].schema) for v in self.visit_list}
        for v, cat in zip(visits, catalogs):
            visit_dict[v].extend(cat)
        # We want catalog contiguity to do object selection later.
        for v in visit_dict:
            visit_dict[v] = visit_dict[v].copy(deep=True)

        return visit_dict

    def _do_plots(self, dataRefs, visitCatalogs,
                  old_relative, old_absolute, new_relative, new_absolute,
                  old_rel_total, old_abs_total, new_rel_total, new_abs_total):
        """Make plots of various quantites to help with debugging."""
        import matplotlib.pyplot as plt
        import astropy.visualization
        # make quantities behave nicely when plotted.
        astropy.visualization.quantity_support()
        plt.ion()

        name = self.id().strip('__main__.')
        old_rms_relative = rms_per_star(old_relative)
        old_rms_absolute = rms_per_star(old_absolute)
        new_rms_relative = rms_per_star(new_relative)
        new_rms_absolute = rms_per_star(new_absolute)
        print("N dataRefs:", len(dataRefs))
        print("relative RMS (old, new):", old_rel_total, new_rel_total)
        print("absolute RMS (old, new):", old_abs_total, new_abs_total)
        plot_rms_histogram(plt, old_rms_relative, old_rms_absolute, new_rms_relative, new_rms_absolute,
                           old_rel_total, old_abs_total, new_rel_total, new_abs_total, name)

        plot_all_wcs_deltas(plt, dataRefs, visitCatalogs, self.oldWcsList, name)


def rms_per_star(data):
    """
    Compute the RMS for each catalog star.

    @param data (dict) dict of starID:list(measurement deltas for that star)

    @return (np.array, astropy.units.arcsecond) the RMS for each starID (no guaranteed order)
    """
    return (np.sqrt(np.array([np.mean(np.array(dd)**2) for dd in data.values()]))*u.radian).to(u.arcsecond)


def rms_total(data):
    """
    Compute the total rms across all stars.

    @param data (dict) dict of starID:list(measurement deltas for that star)

    @return (astropy.units.arcsecond) the total rms across all stars
    """
    total = sum(sum(np.array(dd)**2) for dd in data.values())
    n = sum(len(dd) for dd in data.values())
    return (np.sqrt(total/n) * u.radian).to(u.arcsecond)


def plot_all_wcs_deltas(plt, dataRefs, visitCatalogs, oldWcsList, name,
                        perCcdPlots=False):
    """Various plots of the difference between old and new Wcs."""

    print('plotting heat maps')
    plot_wcs_magnitude(plt, dataRefs, visitCatalogs, oldWcsList, name)
    print('plotting quivers')
    plot_all_wcs_quivers(plt, dataRefs, visitCatalogs, oldWcsList, name)

    if perCcdPlots:
        for i, ref in enumerate(dataRefs):
            md = ref.get('calexp_md')
            plot_wcs(plt, oldWcsList[i], ref.get('wcs').getWcs(),
                     dim=(md.get('NAXIS1'), md.get('NAXIS1')),
                     center=(md.get('CRVAL1'), md.get('CRVAL2')), name='dataRef %d'%i)


def make_xy_wcs_grid(dim, wcs1, wcs2, num=50):
    """Return num x/y grid coordinates for wcs1 and wcs2."""
    x = np.linspace(0, dim[0], num)
    y = np.linspace(0, dim[1], num)
    x1, y1 = wcs_convert(x, y, wcs1)
    x2, y2 = wcs_convert(x, y, wcs2)
    return x1, y1, x2, y2


def wcs_convert(xv, yv, wcs):
    """Convert two arrays of x/y points into an on-sky grid."""
    xout = np.zeros((xv.shape[0], yv.shape[0]))
    yout = np.zeros((xv.shape[0], yv.shape[0]))
    for i, x in enumerate(xv):
        for j, y in enumerate(yv):
            sky = wcs.pixelToSky(x, y).toFk5()
            xout[i, j] = sky.getRa()
            yout[i, j] = sky.getDec()
    return xout, yout


def plot_all_wcs_quivers(plt, dataRefs, visitCatalogs, oldWcsList, name):
    """Make quiver plots of the WCS deltas for each CCD in each visit."""

    for cat in visitCatalogs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for old_wcs, ref in zip(oldWcsList, dataRefs):
            if ref.dataId['visit'] != cat:
                continue
            md = ref.get('calexp_md')
            Q = plot_wcs_quivers(ax, old_wcs, ref.get('wcs').getWcs(),
                                 dim=(md.get('NAXIS1'), md.get('NAXIS2')))
            # TODO: add CCD bounding boxes to plot once DM-5503 is finished.
            # TODO: add a circle for the full focal plane.
        length = (0.1*u.arcsecond).to(u.radian).value
        ax.quiverkey(Q, 0.9, 0.95, length, '0.1 arcsec', coordinates='figure', labelpos='W')
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.title('visit: {}'.format(cat))
        plt.savefig('.plots/{}-{}-quivers.pdf'.format(name, cat))


def plot_wcs_quivers(ax, wcs1, wcs2, dim):
    """Plot the delta between wcs1 and wcs2 as vector arrows."""

    x1, y1, x2, y2 = make_xy_wcs_grid(dim, wcs1, wcs2)
    uu = x2 - x1
    vv = y2 - y1
    return ax.quiver(x1, y1, uu, vv, units='x', pivot='tail', scale=1e-3, width=1e-5)


def plot_wcs_magnitude(plt, dataRefs, visitCatalogs, oldWcsList, name):
    for cat in visitCatalogs:
        fig = plt.figure()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        xmin = np.inf
        ymin = np.inf
        xmax = -np.inf
        ymax = -np.inf
        for old_wcs, ref in zip(oldWcsList, dataRefs):
            if ref.dataId['visit'] != cat:
                continue
            md = ref.get('calexp_md')
            x1, y1, x2, y2 = make_xy_wcs_grid((md.get('NAXIS1'), md.get('NAXIS2')),
                                              old_wcs, ref.get('wcs').getWcs())
            uu = x2 - x1
            vv = y2 - y1
            extent = (x1[0, 0], x1[-1, -1], y1[0, 0], y1[-1, -1])
            xmin = x1.min() if x1.min() < xmin else xmin
            ymin = y1.min() if y1.min() < ymin else ymin
            xmax = x1.max() if x1.max() > xmax else xmax
            ymax = y1.max() if y1.max() > ymax else ymax
            magnitude = (np.linalg.norm((uu, vv), axis=0)*u.radian).to(u.arcsecond).value
            img = ax.imshow(magnitude, vmin=0, vmax=0.3,
                            aspect='auto', extent=extent, cmap=plt.get_cmap('magma'))
            # TODO: add CCD bounding boxes to the plot once DM-5503 is finished.
            # TODO: add a circle for the full focal plane.

        # We're reusing only one of the returned images here for colorbar scaling,
        # but it doesn't matter because we set vmin/vmax so they are all scaled the same.
        cbar = plt.colorbar(img)
        cbar.ax.set_ylabel('distortion (arcseconds)')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.title('visit: {}'.format(cat))
        plt.savefig('.plots/{}-{}-heatmap.pdf'.format(name, cat))


def plot_wcs(plt, wcs1, wcs2, dim, center=(0, 0), name=""):
    """Plot the "distortion map": wcs1-wcs2 delta of points in the CCD grid."""

    plt.figure()

    x1, y1, x2, y2 = make_xy_wcs_grid(dim, wcs1, wcs2, num=50)
    plt.plot((x1 - x2) + center[0], (y1 - y2) + center[1], '-')
    plt.xlabel('delta RA (arcsec)')
    plt.ylabel('delta Dec (arcsec)')
    plt.title(name)


def plot_rms_histogram(plt, old_rms_relative, old_rms_absolute,
                       new_rms_relative, new_rms_absolute,
                       old_rel_total, old_abs_total, new_rel_total, new_abs_total,
                       name):
    """Plot histograms of the star separations and their RMS values."""
    plt.figure()

    color_rel = 'black'
    ls_old = 'dotted'
    color_abs = 'green'
    ls_new = 'dashed'
    plotOptions = {'lw': 2, 'range': (0, 0.1)*u.arcsecond, 'normed': True,
                   'bins': 30, 'histtype': 'step'}

    plt.title('relative vs. absolute: %d vs. %d'%(len(old_rms_relative), len(old_rms_absolute)))

    plt.hist(old_rms_absolute, color=color_abs, ls=ls_old, label='old abs', **plotOptions)
    plt.hist(new_rms_absolute, color=color_abs, ls=ls_new, label='new abs', **plotOptions)

    plt.hist(old_rms_relative, color=color_rel, ls=ls_old, label='old rel', **plotOptions)
    plt.hist(new_rms_relative, color=color_rel, ls=ls_new, label='new rel', **plotOptions)

    plt.axvline(x=old_abs_total.value, linewidth=1.5, color=color_abs, ls=ls_old)
    plt.axvline(x=new_abs_total.value, linewidth=1.5, color=color_abs, ls=ls_new)
    plt.axvline(x=old_rel_total.value, linewidth=1.5, color=color_rel, ls=ls_old)
    plt.axvline(x=new_rel_total.value, linewidth=1.5, color=color_rel, ls=ls_new)

    plt.xlim(plotOptions['range'])
    plt.xlabel('arcseconds')
    plt.legend(loc='best')
    plt.savefig('.plots/%s-histogram.pdf'%name)
