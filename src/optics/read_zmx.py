from zmxtools import zar
import numpy as np
import logging
from optics.glass import Glass

logger_parser = logging.getLogger("parser")

class ZemaxSurface:
    def __init__(self, name=None, parent=None):
        self.name = name
        self.parent = parent
        self.is_stop = False
        self.type = "STANDARD"
        self.curvature = None
        self.conic_constant = None
        self.label = None
        self.distance_to_next = None
        self.diameter = None
        self.mechanical_diameter = None
        self.coating_ref = None
        self.glass = None
        self.parameters = {}

    def apply_command(self, command, params):
        if command == "STOP":
            self.is_stop = True
        elif command == "BLNK":
            pass
        elif command == "TYPE":
            self.type = params
        elif command == "FIMP":
            if params != "":
                raise NotImplementedError("handling non-empty FIMP")
        elif command == "CURV":
            self.curvature = float(params.strip().split(None, 1)[0])
        elif command == "CONI":
            self.conic_constant = float(params.strip().split(None, 1)[0])
        elif command == "MIRR":
            if params != "2 0":
                raise NotImplementedError(f"MIRR with params <{params}>")
        elif command == "SLAB":
            self.label = params
        elif command == "DISZ":
            self.distance_to_next = \
                float(params) if params != "INFINITY" else np.inf
        elif command == "DIAM":
            self.diameter = 2*float(params.strip().split(None, 1)[0])
        elif command == "MEMA":
            self.mechanical_diameter = 2*float(params.strip().split(None, 1)[0])
        elif command == "COAT":
            self.coating_ref = params
        elif command == "FLAP":
            parts = params.strip().split(None, 1)
            ap_type = int(parts[0])
            if ap_type != 0:
                raise NotImplementedError(f"aperture of type {ap_type}")
        elif command == "GLAS":
            parts = params.strip().split(None)
            if parts[0] != "___BLANK":
                glass_ref = parts[0]
                try:
                    self.glass = Glass.from_zemax_data(
                        glass_ref, self.parent)
                except:
                    logger_parser.info(
                        f"{glass_ref} not found in attached files")
                    if glass_ref.startswith("N-"):
                        logger_parser.info(
                            f"stripping initial 'N-' from {glass_ref}")
                        glass_ref = glass_ref[2:]
                    self.glass = Glass.from_library(glass_ref)
            else:
                nd = float(parts[3])
                Vd = float(parts[4])
                self.glass = Glass.from_two_term_model("unnamed", nd, Vd)
        elif command == "PARM":
            parts = params.strip().split(None)
            self.parameters[int(parts[0])] = float(parts[1])
        elif command in ["HIDE", "COMM", "POPS"]:
            logger_parser.info(f"ignoring command <{command}>")
        else:
            logger_parser.info(f"applying unknown command <{command}> "\
                f"with params <{params}> to obj id {id(self)}")

    def __str__(self):
        return \
            f"surface <{self.name}> of type {self.type}:\n"\
            f"  curvature: {self.curvature}\n"\
            f"  conic_constant: {self.conic_constant}\n"\
            f"  diameter: {self.diameter}\n"\
            f"  mechanical_diameter: {self.mechanical_diameter}\n"\
            f"  distance_to_next: {self.distance_to_next}\n"\
            f"  coating: {self.coating_ref}\n"\
            f"  glass: {self.glass.name if self.glass else None}\n"\
            f"  parameters: {self.parameters}\n" + \
            f"  is_stop: {self.is_stop}"

class ZemaxData:
    def __init__(self, filename, keep_only_lens_surfaces=True, encoding=None):
        self.surfaces = []
        self.additional_files = {}
        self.name = None
        self.catalogues = None
        self.length_unit = None
        if filename.lower().endswith(".zmx"):
            with open(filename, "rb") as f:
                self.raw_content = f.read().decode(
                    "utf-16" if encoding is None else encoding)
        else:
            zmx_contents = None
            zmx_additional_files = []
            for f in zar.read(filename):
                if f.file_name.lower().endswith(".zmx"):
                    if zmx_contents is not None:
                        raise ValueError("multiple zmx entries")
                    self.raw_content = f.unpacked_contents.decode(
                        "utf-8" if encoding is None else encoding)
                else:
                    self.additional_files[f.file_name] = f

        for line in self.raw_content.split("\n"):
            self.parse_line(line.strip())

        if keep_only_lens_surfaces:
            self.surfaces = self.surfaces[1:-1]

    def parse_line(self, line):
        if not line:
            return
        parts = line.strip().split(None, 1)
        command = parts[0].upper()
        params = parts[1] if len(parts) > 1 else ""
        if self.surfaces and command != "SURF":
            self.surfaces[-1].apply_command(command, params)
            return
        if command == "VERS":
            self.version = params
        elif command == "NAME":
            self.name = params
        elif command == "UNIT":
            parts = params.strip().split()
            if parts[0] == "MM":
                self.length_unit = 1e-3
            else:
                raise NotImplementedError(f"length unit {parts[0]}")
        elif command == "GCAT":
            self.catalogues = params.strip().split(None)
        elif command == "MODE":
            if params != "SEQ":
                raise NotImplementedError(f"mode type <{params}>")
        elif command == "SURF":
            if int(params) != len(self.surfaces):
                raise ValueError(f"unexpected surface index {params} "\
                    f"while len(surfaces) == {len(self.surfaces)}")
            self.surfaces.append(ZemaxSurface(name=params, parent=self))
        elif command in ["DBDT"]:
            logger_parser.info(f"ignoring line <{line}>")
        else:
            logger_parser.info(f"parsing unknown line <{command}> <{params}>")

    def __str__(self):
        return f"zmx object <{self.name}>:\n" + \
            "\n".join(str(surf) for surf in self.surfaces)
