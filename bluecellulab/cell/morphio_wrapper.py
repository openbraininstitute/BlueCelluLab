"""Morphology loader and HOC code generator built on top of MorphIO.

Provides soma-geometry helpers that mirror NEURON's ``import3d_gui.hoc``
logic, plus :class:`MorphIOWrapper` which loads a morphology from disk
(or from an H5 container), adjusts the soma geometry, and produces the
HOC commands needed to instantiate the cell in NEURON.

Adapted from ``neurodamus/morphio_wrapper.py``.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from numpy.linalg import eig, norm

from bluecellulab.exceptions import BluecellulabError

logger = logging.getLogger(__name__)

# These helpers compute soma geometry the same way NEURON's import3d does.
X, Y, R = 0, 1, 3


def split_morphology_path(morphology_path):
    """Split a morphology path into collection directory, name, and extension.

    Handles two cases:

    - **Regular file on disk** (path exists): uses ``os.path.dirname`` for
      the collection directory and ``os.path.splitext`` for the name and
      extension.
    - **H5 container entry** (path does not exist on disk): walks up via
      ``os.path.dirname`` until an existing filesystem entry (the container)
      is found, then derives the cell name and extension relative to it.

    Args:
        morphology_path: Path to a morphology file, or to a cell entry
            inside an H5 container (e.g. ``/data/merged.h5/cell_name.h5``).

    Returns:
        A tuple ``(collection_dir, morph_name, morph_ext)`` where
        *collection_dir* is the directory or container path, *morph_name*
        is the bare name without extension, and *morph_ext* is the
        file extension (e.g. ``".h5"``).

    Raises:
        BluecellulabError: If no existing filesystem entry is found
            while walking up the path.
    """
    if os.path.exists(morphology_path):
        collection_path = os.path.dirname(morphology_path)
        morph_name, morph_ext = os.path.splitext(os.path.basename(morphology_path))
        return collection_path, morph_name, morph_ext

    collection_path = morphology_path
    while not os.path.exists(collection_path):
        if collection_path == os.path.dirname(collection_path):
            raise BluecellulabError("Failed to split path.")
        collection_path = os.path.dirname(collection_path)

    morph_name, morph_ext = os.path.splitext(os.path.relpath(morphology_path, collection_path))

    return collection_path, morph_name, morph_ext


def contourcenter(xyz):
    """Resample a soma contour and return its centroid.

    Python port of ``contourcenter`` in
    ``lib/hoc/import3d/import3d_sec.hoc``.  Resamples the contour to
    101 evenly-spaced points along its perimeter using linear
    interpolation and returns both the centroid and the resampled points.

    Args:
        xyz: ``(N, 3)`` array of soma contour point coordinates.

    Returns:
        A tuple ``(mean, new_xyz)`` where *mean* is the ``(3,)`` centroid
        of the resampled contour and *new_xyz* is the ``(101, 3)`` array
        of resampled points.
    """
    POINTS = 101

    # np.diff gives N-1 displacement vectors; appending xyz[0] closes the
    # contour by adding the vector from the last point back to the first.
    # The cumulative sum then gives perimeter distances for all N points,
    # and [:-1] drops the duplicate of the starting point (distance 0).
    points = np.vstack((np.diff(xyz[:, [X, Y]], axis=0), xyz[0, [X, Y]]))
    perim = np.cumsum(np.hstack(((0,), norm(points, axis=1))))[:-1]

    d = np.linspace(0, perim[-1], POINTS)
    new_xyz = np.zeros((POINTS, 3))
    for i in range(3):
        new_xyz[:, i] = np.interp(x=d, xp=perim, fp=xyz[:, i])

    mean = np.mean(new_xyz, axis=0)

    return mean, new_xyz


def get_sides(points, major, minor):
    """Split a centred contour into two sides along the major axis.

    Rotates the point array so the maximum major-axis coordinate is last,
    then splits at the minimum into a left and right side (each in
    ascending major-axis order).  Corresponds to line 1191 of
    ``lib/hoc/import3d/import3d_gui.hoc``.

    Args:
        points: ``(N, 2)`` array of contour points centred at the origin.
        major: Unit vector of the major (longest) axis of the ellipsoid
            fitted to the soma contour.
        minor: Unit vector of the minor axis (constrained to the XY plane).

    Returns:
        A tuple ``(sides, rads)`` where each is a length-2 list containing
        the major-axis coordinates and the minor-axis (radial) coordinates
        of the two sides respectively.
    """
    major_coord, minor_coord = np.dot(points, major), np.dot(points, minor)

    imax = np.argmax(major_coord)
    # pylint: disable=invalid-unary-operand-type
    major_coord, minor_coord = (np.roll(major_coord, -imax), np.roll(minor_coord, -imax))

    imin = np.argmin(major_coord)

    sides = [major_coord[:imin][::-1], major_coord[imin:]]
    rads = [minor_coord[:imin][::-1], minor_coord[imin:]]
    return sides, rads


def make_convex(sides, rads):
    """Remove non-convex points from both sides of the contour.

    Enforces monotonicity on each side so that the resulting outline is
    convex.  Points that break convexity are discarded together with
    their corresponding radial values.

    Args:
        sides: Length-2 list of major-axis coordinate arrays for the two
            contour sides, as returned by :func:`get_sides`.
        rads: Length-2 list of minor-axis (radial) coordinate arrays,
            parallel to *sides*.

    Returns:
        ``(sides, rads)`` with non-convex points removed.
    """

    def convex_idx(m):
        """Return a boolean mask selecting elements of *m* that keep it
        convex."""
        idx = np.ones_like(m, dtype=bool)
        last_val = m[-1]
        for i in range(len(m) - 2, -1, -1):
            if m[i] < last_val:
                last_val = m[i]
            else:
                idx[i] = False
        return idx

    for i_side in [0, 1]:
        ci = convex_idx(sides[i_side])
        sides[i_side] = sides[i_side][ci]
        rads[i_side] = rads[i_side][ci]
    return sides, rads


def contour2centroid(mean, points):
    """Convert a soma contour into a stack of cylinders.

    Python port of ``contour2centroid`` in
    ``lib/hoc/import3d/import3d_gui.hoc``.  The algorithm:

    1. Fits an ellipsoid to the centred contour to find the major and
       minor axes via eigen-decomposition.
    2. Splits the contour into two sides along the major axis and
       enforces convexity.
    3. Resamples both sides to 21 evenly-spaced points along the major
       axis and computes cylinder diameters from the two radial profiles.

    Args:
        mean: ``(3,)`` centroid of the contour as returned by
            :func:`contourcenter`.
        points: ``(101, 3)`` resampled contour points (second element
            returned by :func:`contourcenter`).

    Returns:
        A tuple ``(points, diameters)`` where *points* is a ``(21, 3)``
        array of cylinder centre positions along the major axis and
        *diameters* is a ``(21,)`` array of cylinder diameters.
    """
    logging.debug("Converting soma contour into a stack of cylinders")

    # find the major axis of the ellipsoid that best fits the shape
    # assuming (falsely in general) that the center is the mean

    points -= mean
    eigen_values, eigen_vectors = eig(np.dot(points.T, points))

    # To be consistent with NEURON eigen vector directions
    eigen_vectors *= -1

    idx = np.argmax(eigen_values)
    major = eigen_vectors[:, idx]
    # minor is normal and in xy plane
    idx = 3 - np.argmin(eigen_values) - np.argmax(eigen_values)
    minor = eigen_vectors[:, idx]
    minor[2] = 0

    sides, rads = get_sides(points, major, minor)
    sides, rads = make_convex(sides, rads)

    tobj = np.sort(np.hstack(sides))
    new_major_coord = np.linspace(tobj[1], tobj[-2], 21)
    rads[0] = np.interp(new_major_coord, sides[0], rads[0])
    rads[1] = np.interp(new_major_coord, sides[1], rads[1])

    points = major * new_major_coord[:, np.newaxis] + mean
    diameters = np.abs(rads[0] - rads[1])

    # avoid 0 diameter ends
    diameters[0] = np.mean(diameters[:2])
    diameters[-1] = np.mean(diameters[-2:])

    return points, diameters


def _to_sphere(neuron):
    """Replace the soma with a circular contour of equivalent radius.

    Generates 20 evenly-spaced contour points on a circle of the same
    radius as the original soma, centred on the original soma point in
    the XY plane.  Mutates *neuron* in place.

    Args:
        neuron: Mutable MorphIO morphology whose soma will be replaced.
    """
    radius = neuron.soma.diameters[0] / 2.0
    N = 20
    points = np.zeros((N, 3))
    phase = 2 * np.pi / (N - 1) * np.arange(N)
    points[:, 0] = radius * np.cos(phase)
    points[:, 1] = radius * np.sin(phase)
    points += neuron.soma.points[0]
    neuron.soma.points = points
    neuron.soma.diameters = np.repeat(radius, N)


def single_point_sphere_to_circular_contour(neuron):
    """Convert a single-point sphere soma to an equivalent circular contour.

    NEURON's ``import3d_gui.hoc`` converts single-point sphere somas to
    circular contours so that :func:`contour2centroid` can process them
    uniformly.  This function applies the same transformation.

    Args:
        neuron: Mutable MorphIO morphology with a single-point sphere
            soma.  Modified in place.
    """
    logging.debug(
        "Converting 1-point soma (sperical soma) to circular contour representing the same sphere"
    )
    _to_sphere(neuron)


@dataclass
class SectionName:
    """A simple container to uniquely identify a NEURON Section by name and ID.

    Attributes:
        name (str): The name of the section (e.g., "soma", "dend", etc.).
                    This corresponds to the section's logical type or label.
        id (int): The index of the section among all sections with the same name.
                  For example, in a list of dendrites, this would identify
                  dend[0], dend[1], etc.

    Example:
        For NEURON's `soma[0]`, the corresponding SectionName would be:

            SectionName(name="soma", id=0)

        This allows unique referencing even in models where multiple sections
        have the same base name.
    """

    name: str
    id: int

    def __str__(self):
        return f"{self.name}[{self.id}]"


class MorphIOWrapper:
    """Load a MorphIO morphology and generate HOC instantiation commands.

    Given a morphology path (plain file or H5-container entry of the form
    ``container.h5/cell_name``), this class:

    1. Resolves the path via :func:`split_morphology_path`.
    2. Loads the morphology through ``morphio.Collection`` with
       ``Option.nrn_order`` to match NEURON's section ordering.
    3. Adjusts the soma geometry to match ``import3d_gui.hoc``:
       sphere → circular contour; contour → stack of cylinders.
    4. Builds the section-name list and type-ID distribution used to
       emit HOC commands.

    The main entry point is :meth:`morph_as_hoc`, which returns the list
    of HOC commands needed to instantiate the cell in NEURON.
    """

    morph = property(lambda self: self._morph)

    def __init__(self, input_file, options=0):
        """Load and prepare a morphology for HOC usage.

        Args:
            input_file: Path to the morphology.  Either a plain file
                (``cell.asc``, ``cell.swc``, ``cell.h5``) or an H5-container
                entry (``merged.h5/cell_name`` or
                ``merged.h5/cell_name.h5``).
            options: Additional ``morphio.Option`` flags OR-ed into
                ``Option.nrn_order`` when loading.  Defaults to ``0``.
        """
        self._collection_dir, self._morph_name, self._morph_ext = split_morphology_path(input_file)
        self._options = options
        self._build_morph()
        # This logic is similar to what's in BaseCell, but at this point we are still
        # constructing the cell, so we don't yet have access to a fully initialized instance.
        # Therefore, we cannot reuse the BaseCell implementation directly and need
        # a custom solution here.
        self._section_names = self._get_section_names()
        self._build_sec_typeid_distrib()

    def _build_morph(self):
        """Load and finalise the morphology with NEURON-compatible soma
        geometry.

        Loads via ``morphio.Collection`` with ``Option.nrn_order``, then
        recomputes soma points to match ``import3d_gui.hoc``:

        - **Single-point sphere**: converted to a circular contour via
          :func:`single_point_sphere_to_circular_contour`.
        - **Simple contour**: resampled and converted to a cylinder stack
          via :func:`contourcenter` and :func:`contour2centroid`.

        Stores the resulting immutable ``morphio.Morphology`` in
        ``self._morph``.
        """
        try:
            # Lazy import morphio since it has an issue with execl
            from morphio import Collection, Morphology, Option, SomaType
        except ImportError as e:
            raise RuntimeError("MorphIO is not available") from e

        collection = Collection(self._collection_dir, extensions=[self._morph_ext])
        options = self._options | Option.nrn_order
        self._morph = collection.load(self._morph_name, options, mutable=True)

        # Re-compute the soma points as they are computed in import3d_gui.hoc
        if self._morph.soma_type not in {SomaType.SOMA_SINGLE_POINT, SomaType.SOMA_SIMPLE_CONTOUR}:
            msg = f"H5 morphology is not supposed to have a soma of type: {self._morph.soma_type}"
            raise Exception(msg)
        logging.debug(
            "(%s, %s, %s) has soma type : %s",
            self._collection_dir,
            self._morph_name,
            self._morph_ext,
            self._morph.soma_type,
        )

        if self._morph.soma_type == SomaType.SOMA_SINGLE_POINT:
            # See NRN import3d_gui.hoc -> instantiate() -> sphere_rep()
            single_point_sphere_to_circular_contour(self._morph)
        elif self._morph.soma_type == SomaType.SOMA_SIMPLE_CONTOUR:
            # See NRN import3d_gui.hoc -> instantiate() -> contour2centroid()
            mean, new_xyz = contourcenter(self._morph.soma.points)
            self._morph.soma.points, self._morph.soma.diameters = contour2centroid(mean, new_xyz)

        self._morph = Morphology(self._morph)

    def _get_section_names(self) -> list[SectionName]:
        """Build the ordered list of section names for HOC command generation.

        Returns :class:`SectionName` objects in NEURON section order
        (``Option.nrn_order``).  The first entry is always
        ``SectionName("soma", 0)``.

        Each subsequent entry records the section type name (e.g.
        ``"dend"``, ``"axon"``) and its index *within that type group*,
        which is how NEURON addresses sections (e.g. ``dend[0]``,
        ``dend[1]``).

        Returns:
            Ordered list of :class:`SectionName` starting with
            ``SectionName("soma", 0)``.
        """
        result = [SectionName("soma", 0)]

        last_type = None
        type_start_index = 0

        for i, sec in enumerate(self._morph.sections, start=1):
            sec_type = self._morph.section_types[sec.id]

            if sec_type != last_type:
                last_type = sec_type
                type_start_index = i

            relative_index = i - type_start_index
            result.append(SectionName(MorphIOWrapper.type2name(sec_type), relative_index))

        return result

    def _build_sec_typeid_distrib(self):
        """Build a structured array mapping section type IDs to start index and
        count.

        Computes the run-length encoding of ``self._morph.section_types``
        and stores it as a NumPy structured array in
        ``self._sec_typeid_distrib`` with fields ``type_id``, ``start_id``,
        and ``count``.  A synthetic soma row
        ``(type_id=1, start_id=-1, count=1)`` is prepended.

        Example for a morphology with 2724 axon sections then 75 dendrites::

            array([[(1,   -1,    1)],
                   [(2,    0, 2724)],
                   [(3, 2724,   75)]],
                  dtype=[('type_id', '<i8'), ('start_id', '<i8'),
                         ('count', '<i8')])
        """
        self._sec_typeid_distrib = np.dstack(
            np.unique(self._morph.section_types, return_counts=True, return_index=True)
        )[0]
        self._sec_typeid_distrib = np.concatenate(([(1, -1, 1)], self._sec_typeid_distrib), axis=0)
        self._sec_typeid_distrib.dtype = [("type_id", "<i8"), ("start_id", "<i8"), ("count", "<i8")]

    def morph_as_hoc(self):
        """Generate HOC commands to instantiate the cell in NEURON.

        Produces the same output as NEURON's ``import3d_gui.hoc``,
        covering:

        1. ``create`` commands for each section type and its SectionList
           subset (``somatic``, ``axonal``, ``basal``, ``apical``).
        2. ``forall all.append`` to populate the global ``all`` SectionList.
        3. ``pt3dadd`` calls for soma points (in reverse order to match
           NEURON's convention).
        4. ``connect`` and ``pt3dadd`` calls for every other section.

        Returns:
            List of HOC command strings, each executable via
            ``neuron.h(cmd)`` or joinable into a single block.
        """
        cmds = []
        # Generate create commands for each section type.
        # E.g.: ( soma , 1  ) ( dend , 52 ) ( axon , 23 ) ( apic , 5  )
        for [(type_id, count)] in self._sec_typeid_distrib[["type_id", "count"]]:
            tstr = self.type2name(type_id)
            tstr1 = f"create {tstr}[{count}]"
            cmds.append(tstr1)
            tstr1 = self.mksubset(type_id, tstr)
            cmds.append(tstr1)

        cmds.append("forall all.append")

        # generate 3D soma points commands. Order is reversed wrt NEURON's soma points.
        cmds.extend(
            (
                f"soma {{ pt3dadd({p[0]:.8g}, {p[1]:.8g}, {p[2]:.8g}, {d:.8g}) }}"
                for p, d in zip(
                    reversed(self._morph.soma.points),
                    reversed(self._morph.soma.diameters),
                    strict=True,
                )
            )
        )

        # generate sections connect + their respective 3D points commands
        for i, sec in enumerate(self._morph.sections):
            index = i + 1
            tstr = self._section_names[index]

            if not sec.is_root:
                if sec.parent is not None:
                    parent_index = sec.parent.id + 1
                    tstr1 = self._section_names[parent_index]
                    tstr1 = f"{tstr1} connect {tstr}(0), {1}"
                    cmds.append(tstr1)
            else:
                tstr1 = f"soma connect {tstr}(0), {0.5}"
                cmds.append(tstr1)

                # pt3dstyle does not impact simulation numbers. This will be kept for x-reference.
                # tstr1 = "{} {{ pt3dstyle(1, {:.8g}, {:.8g}, {:.8g}) }}".format
                #                   (tstr, mean[0], mean[1], mean[2])
                # cmds.append(tstr1)

            # 3D point info
            cmds.extend(
                f"{tstr} {{ pt3dadd({p[0]:.8g}, {p[1]:.8g}, {p[2]:.8g}, {d:.8g}) }}"
                for p, d in zip(sec.points, sec.diameters, strict=True)
            )

        return cmds

    # [START] Python equivalents of import3d_gui.hoc helper functions.
    # Original HOC function names are kept for cross-reference.

    _type2name_dict = {1: "soma", 2: "axon", 3: "dend", 4: "apic"}

    @classmethod
    def type2name(cls, type_id):
        """Return the HOC section-name string for a MorphIO type ID.

        Args:
            type_id: Integer section-type identifier
                (1 = soma, 2 = axon, 3 = dend, 4 = apic).

        Returns:
            Section-name string from the standard mapping, or
            ``"minus_<n>"`` for negative type IDs and
            ``"dend_<n>"`` for unknown positive ones.
        """
        return cls._type2name_dict.get(type_id) or (
            f"minus_{-type_id}" if type_id < 0 else f"dend_{type_id}"
        )

    _mksubset_dict = {1: "somatic", 2: "axonal", 3: "basal", 4: "apical"}

    @classmethod
    def mksubset(cls, type_id, type_name):
        """Return the HOC command that appends sections to their subset list.

        Args:
            type_id: Integer section-type identifier
                (1 = soma, 2 = axon, 3 = dend, 4 = apic).
            type_name: HOC section-name string as returned by
                :meth:`type2name`.

        Returns:
            HOC command string of the form
            ``'forsec "<type_name>" <subset>.append'``.
        """
        tstr = cls._mksubset_dict.get(type_id) or (
            f"minus_{-type_id}set" if type_id < 0 else f"dendritic_{type_id}"
        )

        tstr1 = f'forsec "{type_name}" {tstr}.append'
        return tstr1
