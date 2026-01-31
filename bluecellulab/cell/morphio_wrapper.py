"""A wrapper for MorphIO objects.

Provides additional neuron features on top of MorphIO basic morphology
handling. Adapted from neurodamus/morphio_wrapper.py
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from numpy.linalg import eig, norm

from bluecellulab.exceptions import BluecellulabError

logger = logging.getLogger(__name__)
"""[START] Implementations retrieved from nse/morph-tool (!= hpc/morpho-tool
!!!) These functions are needed for soma points computation (a la import3d).

TODO: Once we have something stable, integrate nse/morph-tool
"""
X, Y, R = 0, 1, 3


def split_morphology_path(morphology_path):
    """Split `{collection_path}/{morph_name}.{ext}`"""
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
    """Python implementation of NEURON code: lib/hoc/import3d/import3d_sec.hoc"""
    POINTS = 101

    points = np.vstack((np.diff(xyz[:, [X, Y]], axis=0), xyz[0, [X, Y]]))
    perim = np.cumsum(np.hstack(((0,), norm(points, axis=1))))[:-1]

    d = np.linspace(0, perim[-1], POINTS)
    new_xyz = np.zeros((POINTS, 3))
    for i in range(3):
        new_xyz[:, i] = np.interp(x=d, xp=perim, fp=xyz[:, i])

    mean = np.mean(new_xyz, axis=0)

    return mean, new_xyz


def get_sides(points, major, minor):
    """Circular permutation of the points so that the point with the largest
    coordinate along the major axis becomes the last point tobj =
    major.c.mul(d.x[i])  ###### uneeded?

    line 1191
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
    """Keep only points that make path convex."""

    def convex_idx(m):
        """Return index to elements of m that make it convex.

        Note: not efficient at the moment
        # now we have the two sides without the min and max points (rads[0]=0)
        # we hope both sides now monotonically increase, i.e. convex
        # make it convex
        """
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
    """This follows the function in lib/hoc/import3d/import3d_gui.hoc most of
    the comments are from there, so if you want to follow along, it should
    break up the function the same way."""
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
    """Convert a 3-pts cylinder or a 1-pt sphere into a circular contour that
    represents the same sphere."""
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
    """Transform a single point soma that represents a sphere into a circular
    contour that represents the same sphere."""
    logging.debug(
        "Converting 1-point soma (sperical soma) to circular contour representing the same sphere"
    )
    _to_sphere(neuron)


"""
    [END] Implementations retrieved from nse/morph-tool (!= hpc/morpho-tool !!!)
"""


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
    """A class that wraps a MorphIO object and gets everything ready for HOC
    usage."""

    morph = property(lambda self: self._morph)

    def __init__(self, input_file, options=0):
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
        """Build immutable morphology, going trough mutable and applying neuron
        adjustemnts."""
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
            """ See NRN import3d_gui.hoc -> instantiate()
                            -> sphere_rep(xx, yy, zz, dd) """
            single_point_sphere_to_circular_contour(self._morph)
        elif self._morph.soma_type == SomaType.SOMA_SIMPLE_CONTOUR:
            """ See NRN import3d_gui.hoc -> instantiate()
                            -> contour2centroid(xx, yy, zz, dd, sec) """
            mean, new_xyz = contourcenter(self._morph.soma.points)
            self._morph.soma.points, self._morph.soma.diameters = contour2centroid(mean, new_xyz)

        self._morph = Morphology(self._morph)

    def _get_section_names(self) -> list[SectionName]:
        """Returns a list of SectioName.

        Relative_index is the index of the section within its type
        group, as expected by the NEURON simulator. NEURON organizes
        mechanisms and pointers (e.g., for synapses or diffusion) into
        arrays grouped by section type (e.g., axon, dendrite), so this
        relative index identifies the section position within its group.
        The list starts with ('soma', 0).
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
        """Build typeid distribution on top of MorphIO section_types."""
        """This will hold np.array that will map the different type ids to the
        start id wrt to section_types and their count. For example, axon
        type(2) starts at section.id 0 and and totals 2724 sections.

                | type_id | start_id  |   count   |
                | ------- | --------- | --------- |
                |      2  |        0  |     2724  |
                |      3  |     2724  |       75  |


            >>> _sec_typeid_distrib
            array([[(2,    0, 2724)],
                   [(3, 2724,   75)]],
                  dtype=[('type_id', '<i8'), ('start_id', '<i8'), ('count', '<i8')])

            Then use it like this:
            >>> _sec_typeid_distrib[['type_id', 'start_id']]
            array([[(2,    0)],
                   [(3, 2724)]],
                  dtype={'names':['type_id','start_id'],....}
        """
        self._sec_typeid_distrib = np.dstack(
            np.unique(self._morph.section_types, return_counts=True, return_index=True)
        )[0]
        self._sec_typeid_distrib = np.concatenate(([(1, -1, 1)], self._sec_typeid_distrib), axis=0)
        self._sec_typeid_distrib.dtype = [("type_id", "<i8"), ("start_id", "<i8"), ("count", "<i8")]

    def morph_as_hoc(self):
        """Uses morphio object to read and generate hoc commands just like
        import3d_gui.hoc."""
        cmds = []
        """We need to get the number of sections for each type in order to
        generate the create commands.

        E.g.: ( soma , 1  ) ( dend , 52 ) ( axon , 23 ) ( apic , 5  )
        """
        # generate create commands
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

    """
         [START] Python versions of import3d_gui.hoc helper functions
         Note: nrn function names will be kept for reference
    """

    _type2name_dict = {1: "soma", 2: "axon", 3: "dend", 4: "apic"}

    @classmethod
    def type2name(cls, type_id):
        """:param type_id: id of section type
        :return: name representation of the section type.
                 If not found in _type2name_dict, then default is:
                    if (type < 0): "minus_{}".format(-type)
                    else: "dend_{}".format(type)
        """
        return cls._type2name_dict.get(type_id) or (
            f"minus_{-type_id}" if type_id < 0 else f"dend_{type_id}"
        )

    _mksubset_dict = {1: "somatic", 2: "axonal", 3: "basal", 4: "apical"}

    @classmethod
    def mksubset(cls, type_id, type_name):
        """:param type_id: id of section type
        :param type_name: the name of the section type
        :return: command to append section type to subset
        """
        tstr = cls._mksubset_dict.get(type_id) or (
            f"minus_{-type_id}set" if type_id < 0 else f"dendritic_{type_id}"
        )

        tstr1 = f'forsec "{type_name}" {tstr}.append'
        return tstr1
