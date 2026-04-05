from zmxtools import zar
import numpy as np
import logging

logger_parser = logging.getLogger("parser")
logging.basicConfig(level="INFO")

class ZemaxSurface:
    def __init__(self):
        self.is_stop = False
        self.type = None
        self.curvature = None
        self.label = None
        self.distance_to_next = None
        self.diameter = None
        self.mechanical_diameter = None
        self.coating_ref = None
        self.glass_ref = None
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
                self.glass_ref = parts[0]
            else:
                self.glass_ref = parts
        elif command == "PARM":
            parts = params.strip().split(None)
            self.parameters[int(parts[0])] = float(parts[1])
        elif command in ["HIDE", "COMM", "POPS"]:
            logger_parser.info(f"ignoring command <{command}>")
        else:
            logger_parser.info(f"applying unknown command <{command}> "\
                f"with params <{params}> to obj id {id(self)}")

    def __str__(self):
        return f"surface <{self.label}> of type {self.type} "\
            f"with curvature {self.curvature}, diameter {self.diameter}, "\
            f"mechanical-diameter {self.mechanical_diameter} and "\
            f"distance-to-next {self.distance_to_next} and "\
            f"coating {self.coating_ref} and glass {self.glass_ref} " + \
            f"and parameters {self.parameters}" + \
            (" (STOP)" if self.is_stop else "")

class ZemaxData:
    def __init__(self, filename, keep_only_lens_surfaces=True):
        self.surfaces = []
        if filename.lower().endswith(".zmx"):
            with open(filename, "rb") as f:
                zmx_contents = f.read()
            zmx_additional_files = []
        else:
            zmx_contents = None
            zmx_additional_files = []
            for f in zar.read(filename):
                if f.file_name.lower().endswith(".zmx"):
                    if zmx_contents is not None:
                        raise ValueError("multiple zmx entries")
                    zmx_contents = f.unpacked_contents
                else:
                    zmx_additional_files.append(f)

        self.raw_content = zmx_contents.decode("utf-8")
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
        elif command == "MODE":
            if params != "SEQ":
                raise NotImplementedError(f"mode type <{params}>")
        elif command == "SURF":
            if int(params) != len(self.surfaces):
                raise ValueError(f"unexpected surface index {params} "\
                    f"while len(surfaces) == {len(self.surfaces)}")
            self.surfaces.append(ZemaxSurface())
        elif command in ["DBDT"]:
            logger_parser.info(f"ignoring line <{line}>")
        else:
            logger_parser.info(f"parsing unknown line <{command}> <{params}>")

    def __str__(self):
        return "zmx object consisting of the following surfaces:\n" + \
            "\n".join(str(surf) for surf in self.surfaces)
