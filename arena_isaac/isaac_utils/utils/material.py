from __future__ import annotations

import functools
import re
from pathlib import Path
import tempfile

import omni
from isaac_utils.utils.path import world_path

from isaacsim_msgs.msg import Material as MaterialMsg


class MdlPreprocessor:
    @classmethod
    def relative_to_absolute(cls, base_path: Path, relative_path: str) -> str:
        """Convert a relative MDL path to an absolute path.

        Args:
            base_path (Path): Base path to resolve relative paths against.
            relative_path (str): Relative path.

        Returns:
            str: Absolute path from the base path.
        """
        abs_path = (base_path / relative_path).resolve()
        return str(abs_path)

    @classmethod
    def preprocess_mdl(cls, mdl_path: Path) -> Path | None:
        """Preprocess an MDL file to resolve relative paths.

        Args:
            mdl_path (Path): Path to the MDL file.

        Returns:
            Path | None: Path to the preprocessed MDL file, or None if no substitutions performed or processing failed.
        """
        if not mdl_path.exists():
            return None

        base_path = mdl_path.parent

        with open(mdl_path, 'r') as f:
            mdl_content = f.read()

        replacer_fn = functools.partial(cls.relative_to_absolute, base_path)
        mdl_content, subs = re.subn(r'"(\./[^"]+)"', lambda m: f'"{replacer_fn(m.group(1))}"', mdl_content)

        if not subs:
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{mdl_path.name}', mode='w') as tmp_file:
            tmp_file.write(mdl_content)
            return Path(tmp_file.name)


class Material:
    _path: str

    @classmethod
    def from_msg(cls, msg: MaterialMsg) -> Material | None:
        """
        Create a material from a MaterialMsg.
        :param msg: The msg to create the material from.
        :return: The created material, or None if the material could not be created.
        """
        return cls.load(path=Path(msg.path), name=msg.name) if msg.path and msg.name else None

    @classmethod
    def load(cls, path: Path, name: str) -> Material | None:
        """
        Load a material from an MDL path.
        :param path: The MDL path of the material to load.
        :param name: The name of the material to load.
        :return: The loaded material, or None if the material could not be loaded.
        """
        material_path = world_path('Looks', 'Material', f'{name}_{str(hash(path))[8:16]}')

        basename, *ref = path.name.split('::', 1)

        if (processed := MdlPreprocessor.preprocess_mdl(path.parent / basename)) is not None:
            path = processed.parent / (processed.name + (f'::{ref[0]}' if ref else ''))

        stage = omni.usd.get_context().get_stage()
        mtl = stage.GetPrimAtPath(material_path)

        if not (mtl and mtl.IsValid()):
            if not omni.kit.commands.execute(
                'CreateMdlMaterialPrimCommand',
                mtl_url=str(path),
                mtl_name=name,
                mtl_path=material_path
            ):
                return None

        obj = cls()
        obj._path = material_path
        return obj

    @property
    def path(self) -> str:
        return self._path

    def bind_to(self, prim_path: str) -> bool:
        """
        Bind this material to a prim.
        :param prim_path: The path of the prim to bind the material to.
        :return: True if the material was successfully bound, False otherwise.
        """
        return omni.kit.commands.execute(
            'BindMaterialCommand',
            prim_path=prim_path,
            material_path=self._path
        )
