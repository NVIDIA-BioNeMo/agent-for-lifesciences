# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandboxed Python execution environment.

Provides secure code execution with restricted access to system resources
while allowing scientific computing with numpy, scipy, and rdkit.

Supports the Aevom data store integration, allowing code to load stored
molecular data (PDB, SDF, selections) via the `aevom` client.
"""

import ast
import sys
import io
import signal
import tokenize
import traceback
import types
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Optional
from RestrictedPython import compile_restricted, safe_builtins, PrintCollector
from RestrictedPython.Guards import (
    safe_globals,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.transformer import RestrictingNodeTransformer


class ScientificComputingTransformer(RestrictingNodeTransformer):
    """Custom RestrictedPython transformer optimized for scientific computing.
    
    This transformer relaxes certain restrictions that are overly burdensome
    for scientific computing code while maintaining security-critical guards.
    
    Relaxed restrictions:
    - Augmented assignments (+=, -=, *=, etc.) on simple names
    - `nonlocal` statements (useful for closures)
    - Function calls with *args/**kwargs (common in scientific APIs)
    
    Maintained restrictions:
    - Attribute access guarding (_getattr_, _write_)
    - Subscript access guarding (_getitem_)
    - Iteration guarding (_getiter_)
    - Protected name access (names starting with _)
    - exec/eval prohibition
    - Import restrictions (handled separately by _guarded_import)
    """
    
    def visit_AugAssign(self, node):
        """Allow augmented assignments without transformation.
        
        The default RestrictedPython behavior transforms:
            x += 1  ->  x = _inplacevar_('+=', x, 1)
        
        This requires providing an _inplacevar_ guard function.
        
        For scientific computing, augmented assignments are extremely common
        and safe for simple variable names. We allow them to pass through
        unchanged, which means:
            x += 1  stays as  x += 1
        
        Note: RestrictedPython's original implementation also forbids:
        - obj.attr += 1 (augmented assignment of attributes)
        - list[0] += 1 (augmented assignment of subscripts)
        
        We inherit this safety by only allowing Name targets through.
        For Attribute and Subscript targets, we still forbid them.
        """
        import ast
        node = self.node_contents_visit(node)
        
        # Forbid augmented assignment of attributes (security concern)
        if isinstance(node.target, ast.Attribute):
            self.error(
                node,
                "Augmented assignment of attributes is not allowed.")
            return node
        
        # Forbid augmented assignment of subscripts (security concern). The
        # message teaches the fix because this is the single most common thing
        # models get wrong in the sandbox -- counting into a dict with
        # ``counts[k] += 1`` -- and a bare rejection just gets retried the same
        # way. Plain item assignment (``d[k] = v``) is allowed; only the
        # augmented form on a subscript is not.
        if isinstance(node.target, ast.Subscript):
            self.error(
                node,
                "Augmented assignment of object items and slices is not allowed "
                "(e.g. counts[k] += 1, arr[i] += x). Use counts[k] = "
                "counts.get(k, 0) + 1, or collections.Counter for counting, or "
                "build the value with a comprehension -- plain d[k] = v is fine.")
            return node
        
        # Allow augmented assignment of simple names (x += 1)
        # Just return the node unchanged - no _inplacevar_ transformation
        return node
    
    def visit_Nonlocal(self, node):
        """Allow `nonlocal` statements.
        
        The default RestrictedPython behavior forbids `nonlocal` entirely.
        
        For scientific computing, `nonlocal` is useful for closures and
        callbacks, which are common patterns. The security risk is minimal
        since nonlocal only affects the enclosing scope within the same
        code execution context.
        """
        # Just visit and allow - don't call self.not_allowed(node)
        return self.node_contents_visit(node)
    
    def visit_Call(self, node):
        """Allow function calls with *args and **kwargs without transformation.
        
        The default RestrictedPython behavior transforms:
            func(*args, **kwargs)  ->  _apply_(func, *args, **kwargs)
        
        This requires providing an _apply_ guard function.
        
        For scientific computing, *args and **kwargs are extremely common
        (numpy, scipy, rdkit all use them extensively). We allow them
        to pass through unchanged.
        
        We still maintain the prohibition on exec() and eval().
        """
        import ast
        
        # Still forbid exec and eval
        if isinstance(node.func, ast.Name):
            if node.func.id == 'exec':
                self.error(node, 'Exec calls are not allowed.')
            elif node.func.id == 'eval':
                self.error(node, 'Eval calls are not allowed.')
        
        # Visit child nodes but don't wrap in _apply_
        return self.node_contents_visit(node)


def _print_factory():
    """Create a print collector that mimics print() behavior.
    
    RestrictedPython transforms print() calls to use _print_().
    We need to provide a _print_ factory that returns a callable.
    The PrintCollector accumulates output and stores it in 'printed'.
    """
    return PrintCollector

from .aevom_client import AevomClient, create_client


class TimeoutError(Exception):
    """Raised when code execution exceeds the time limit."""
    pass


class SandboxError(Exception):
    """Raised when sandbox security is violated."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutError("Code execution timed out")


# Submodules of otherwise-allowed packages that reopen the file and network
# access the sandbox exists to prevent. Both ways in are closed: the import
# statement, in _guarded_import, and attribute traversal from the parent package,
# in _guarded_getattr. The second is not hypothetical -- sklearn resolves
# `sklearn.datasets` lazily on attribute access, so blocking only the import
# would leave fetch_openml() one dot away.
BLOCKED_SUBMODULES = {
    "sklearn": frozenset(
        {
            # fetch_openml() and friends download over HTTP; load_svmlight_file()
            # and dump_svmlight_file() read and write arbitrary paths.
            "datasets",
            # Vendored third-party packages, historically including joblib. What
            # is behind it changes between releases, so it is not worth auditing.
            "externals",
        }
    ),
    "biotite": frozenset(
        {
            # fetch() against RCSB, UniProt, Entrez and PubChem -- the network
            # access the sandbox exists to prevent. Structures reach sandbox code
            # through aevom.load_pdb() instead.
            "database",
            # Wrappers that launch external executables (DSSP, MUSCLE, BLAST).
            # Nothing here is installed in the image, but the point is that these
            # spawn processes, which is what the subprocess ban already forbids.
            "application",
        }
    ),
}

_BLOCKED_MODULE_NAMES = frozenset(
    f"{package}.{module}"
    for package, modules in BLOCKED_SUBMODULES.items()
    for module in modules
)

_BLOCKED_MODULE_MESSAGE = (
    "Import of '{name}' is not allowed in sandbox because it can read files, "
    "fetch over the network, or launch external programs. Use the aevom client "
    "for data access; the rest of the package is available for analysis."
)


def _guarded_import(name, globalz=None, localz=None, fromlist=(), level=0):
    """Restricted import that only allows whitelisted modules."""
    ALLOWED_MODULES = {
        # Standard library - safe subset
        "math",
        "statistics",
        "collections",
        "itertools",
        "functools",
        "operator",
        "json",
        "re",
        "copy",
        "decimal",
        "fractions",
        "io",  # StringIO for BioPython PDB parsing
        # Scientific computing
        "numpy",
        "scipy",
        "scipy.spatial",
        "scipy.spatial.distance",
        "scipy.stats",
        "scipy.optimize",
        "scipy.linalg",
        # BioPython for PDB parsing
        "Bio",
        "Bio.PDB",
        "Bio.PDB.PDBParser",
        "Bio.PDB.NeighborSearch",
        "Bio.PDB.Selection",
        "Bio.PDB.Polypeptide",
        "Bio.PDB.vectors",
        # Chemistry (RDKit submodules)
        # Note that "rdkit" alone admits the whole package, since the check
        # below matches on module prefixes. The submodules are listed for
        # discoverability, not because they would otherwise be blocked.
        "rdkit",
        "rdkit.Chem",
        "rdkit.Chem.AllChem",
        "rdkit.Chem.Descriptors",
        "rdkit.Chem.rdMolDescriptors",
        "rdkit.Chem.Lipinski",
        "rdkit.Chem.Crippen",
        "rdkit.Chem.MolSurf",
        "rdkit.Chem.Scaffolds.MurckoScaffold",
        "rdkit.Chem.rdFMCS",
        "rdkit.Chem.FilterCatalog",
        "rdkit.Chem.ChemicalFeatures",
        "rdkit.Chem.BRICS",
        "rdkit.DataStructs",
        # File-format conversion and forcefield work: OBConversion, OBMol,
        # AddHydrogens, Gasteiger charges. Reached as `from openbabel import
        # openbabel` or `pybel`. Read and write through ReadString/WriteString --
        # the path-based calls have nothing to point at in the sandbox.
        "openbabel",
        # Structure parsing, superimposition, SASA and secondary structure over
        # AtomArrays. See BLOCKED_SUBMODULES for the network and subprocess parts
        # that stay out.
        "biotite",
        # Clustering, dimensionality reduction and regression over molecular
        # descriptors. See BLOCKED_MODULES for the parts that stay out.
        "sklearn",
        # SAFE fragment notation (safe.encode / safe.decode). Installed with a
        # lazy __init__ in the RDKit base image, so safe.tokenizer and friends
        # raise ImportError rather than dragging in torch.
        "safe",
        # PubChem API access (for molecule search)
        "pubchempy",
        # Sandbox-specific modules (registered in sys.modules at runtime)
        "aevom",
    }
    
    # Checked before the prefix match below, which would otherwise admit these
    # along with their parent package.
    if name in _BLOCKED_MODULE_NAMES or any(
        name.startswith(f"{blocked}.") for blocked in _BLOCKED_MODULE_NAMES
    ):
        raise ImportError(_BLOCKED_MODULE_MESSAGE.format(name=name))

    # `from sklearn import datasets` asks to import "sklearn" and pulls the
    # submodule out of the fromlist, so the name check above never sees it.
    blocked_here = BLOCKED_SUBMODULES.get(name, frozenset())
    for requested in fromlist or ():
        if requested in blocked_here:
            raise ImportError(
                _BLOCKED_MODULE_MESSAGE.format(name=f"{name}.{requested}")
            )

    # datamol is reachable as a re-export even though it is not whitelisted:
    # `safe.converter` holds a module-level `dm`, so `from safe.converter import
    # dm` would hand the sandbox datamol's fsspec-backed IO -- dm.read_csv() and
    # dm.to_sdf() take URLs and filesystem paths, which is exactly what the
    # `open(` and `socket.` guards exist to prevent. Blocked here rather than in
    # validate_code because the pattern list cannot see through an alias.
    if "dm" in (fromlist or ()):
        raise ImportError(
            "Importing 'dm' (datamol) is not allowed in sandbox. Use rdkit.Chem "
            "for molecule handling, or the aevom client for file and network access."
        )

    # Check if the module or any parent module is allowed
    module_parts = name.split(".")
    for i in range(len(module_parts)):
        check_name = ".".join(module_parts[:i+1])
        if check_name in ALLOWED_MODULES:
            return __builtins__["__import__"](name, globalz, localz, fromlist, level)
    
    raise ImportError(f"Import of '{name}' is not allowed in sandbox")


def _guarded_getattr(obj, name):
    """Restricted getattr that prevents access to dangerous attributes."""
    # The attribute-traversal half of the datamol block above: `safe.converter.dm`
    # reaches the same IO surface without an import statement. Scoped to modules
    # so a user's own object may still have a `dm` attribute.
    if name == "dm" and isinstance(obj, types.ModuleType):
        raise SandboxError(
            "Access to 'dm' (datamol) is not allowed in sandbox. Use rdkit.Chem "
            "for molecule handling, or the aevom client for file and network access."
        )

    # The traversal half of the BLOCKED_SUBMODULES block. Scoped to the parent
    # package so an unrelated object may still have an attribute of that name.
    if isinstance(obj, types.ModuleType):
        package = getattr(obj, "__name__", "")
        if name in BLOCKED_SUBMODULES.get(package, frozenset()):
            raise SandboxError(
                _BLOCKED_MODULE_MESSAGE.format(name=f"{package}.{name}").replace(
                    "Import of", "Access to"
                )
            )

    FORBIDDEN_ATTRS = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__dict__",
        "__delattr__",
        "__reduce__",
        "__reduce_ex__",
        "__getattribute__",
        "__setattr__",
    }
    
    if name.startswith("_") and name in FORBIDDEN_ATTRS:
        raise SandboxError(f"Access to '{name}' is not allowed")
    
    return getattr(obj, name)


def _write_guard(obj):
    """Guard function for write operations."""
    # Allow writing to standard containers
    if isinstance(obj, (list, dict, set)):
        return obj
    # Disallow other write operations
    raise SandboxError("Write operations are restricted")


def sanitize_code(code: str) -> str:
    """Strip markdown code fencing that LLMs sometimes include in tool calls.
    
    LLMs often generate code with markdown fencing like:
    ```python
    print("hello")
    ```
    
    This function removes the fencing so the code can be executed.
    
    Args:
        code: Python code string, possibly with markdown fencing.
        
    Returns:
        Clean Python code without markdown fencing.
    """
    if not code:
        return code
    
    lines = code.strip().split('\n')
    
    # Remove leading ```python or ``` (with optional language specifier)
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    
    # Remove trailing ```
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    
    return '\n'.join(lines)


@dataclass
class SandboxContext:
    """Context for sandbox execution with data store access.
    
    Attributes:
        session_token: Token for authenticating with backend data store.
        backend_url: Base URL of the backend API.
        allowed_refs: Set of reference IDs this session can access.
    """
    session_token: str
    backend_url: str
    allowed_refs: set[str]
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["SandboxContext"]:
        """Create SandboxContext from a dict (e.g., from JSON request).
        
        Args:
            data: Dict with session_token, backend_url, and allowed_refs.
            
        Returns:
            SandboxContext or None if data is empty/invalid.
        """
        if not data:
            return None
        
        session_token = data.get("session_token")
        backend_url = data.get("backend_url")
        allowed_refs = data.get("allowed_refs", [])
        
        if not session_token or not backend_url:
            return None
        
        return cls(
            session_token=session_token,
            backend_url=backend_url,
            allowed_refs=set(allowed_refs),
        )


def create_sandbox_globals(
    context: dict[str, Any] | None = None,
    sandbox_context: SandboxContext | None = None,
) -> dict[str, Any]:
    """Create the globals dict for sandboxed execution.
    
    Args:
        context: Additional context variables to inject (e.g., selection data).
        sandbox_context: Optional sandbox context for data store access.
        
    Returns:
        Globals dict with safe builtins and allowed modules.
    """
    # Start with safe builtins
    sandbox_globals = {
        "__builtins__": {
            **safe_builtins,
            "__import__": _guarded_import,
            # Note: print is handled via _print_ which uses PrintCollector
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            "all": all,
            "any": any,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": _guarded_getattr,
            "dir": dir,  # Safe: only returns list of attribute names
            "format": format,
            "repr": repr,
            "next": next,
            "iter": iter,
            "frozenset": frozenset,
            "bin": bin,
            "ascii": ascii,
            "object": object,
            "super": super,
            "classmethod": classmethod,
            "staticmethod": staticmethod,
            "property": property,
            "bytearray": bytearray,
            "memoryview": memoryview,
            "__metaclass__": type,
        },
        "_getattr_": _guarded_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": iter,
        # The two guards are not interchangeable: the iter_ variant is a
        # generator that unpacks each element of an iterable, for `for a, b in x`.
        # Using it for plain `a, b = x` unpacks the elements of x instead of x
        # itself, so `a, b = ('x', 1.0)` dies on "'float' object is not iterable".
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_": _write_guard,
        # Print support - RestrictedPython transforms print() to use _print_()
        # The PrintCollector class is passed; it's instantiated per-execution as _print in locals
        "_print_": PrintCollector,
        "__name__": "__sandbox__",
    }
    
    # Pre-import commonly used modules so they're available
    try:
        import numpy as np
        import math
        sandbox_globals["np"] = np
        sandbox_globals["numpy"] = np
        sandbox_globals["math"] = math
    except ImportError:
        pass
    
    # Pre-load RDKit Fragment patterns (they require calling _LoadPatterns which sandbox blocks)
    try:
        from rdkit.Chem import Fragments
        Fragments._LoadPatterns()
    except Exception:
        pass  # Silently ignore if RDKit data files not available
    
    # Create and inject the aevom client if sandbox context is provided
    if sandbox_context:
        aevom_client = create_client(
            session_token=sandbox_context.session_token,
            backend_url=sandbox_context.backend_url,
            allowed_refs=sandbox_context.allowed_refs,
        )
        sandbox_globals["aevom"] = aevom_client
        sys.modules["aevom"] = aevom_client  # Enable: import aevom
    
    # Add context variables
    if context:
        sandbox_globals.update(context)
        # The injected selection gets the same residue helpers as one loaded
        # with aevom.load_selection(); it is still a dict in every other way.
        if isinstance(sandbox_globals.get("selection"), dict):
            from .aevom_client import Selection

            sandbox_globals["selection"] = Selection(sandbox_globals["selection"])
    
    return sandbox_globals


# Bare builtin calls that are dangerous. An *attribute* of the same name is
# fine and common (``re.compile(...)``, ``np.load(...)``, ``obj.open()``), so
# these only trip when called as a bare name -- i.e. not preceded by a dot.
_FORBIDDEN_BARE_CALLS = frozenset({
    "exec", "eval", "compile", "open", "file", "__import__",
    "globals", "locals", "vars", "setattr", "delattr",
})
# Identifiers for dangerous modules. RestrictedPython already blocks importing
# them; this is a second, precise line that also rejects them as an aliased
# attribute base. Matched as a whole token and only when not itself an
# attribute, so ``mysocket``, ``unpickle`` and ``df.socket`` do not trip it.
_FORBIDDEN_NAMES = frozenset({
    "subprocess", "socket", "pickle", "marshal", "shelve", "pty",
    "commands", "builtins",
})
# Dangerous attribute accesses keyed by their base identifier (``os.system``).
_FORBIDDEN_ATTRS = {"os": frozenset({"system", "popen"})}


def validate_code(code: str) -> tuple[bool, str | None]:
    """Validate code for obvious security issues before compilation.

    The real sandboxing is done by RestrictedPython -- guarded builtins,
    guarded ``getattr``/``getitem``, and a whitelisted import hook. This is a
    fast, friendly pre-check that rejects the most obvious escape attempts with
    a clear message instead of a downstream ``NameError``.

    It scans *tokens*, not raw text. The previous implementation did a
    case-folded substring search, which flagged legitimate code: ``file(``
    matched identifiers like ``structure_file(``, ``compile(`` matched
    ``re.compile(``, and every pattern matched inside strings and comments. A
    dangerous name inside a string, a comment, or embedded in a longer
    identifier is not a call to the dangerous thing, so token-level matching
    removes that whole class of false positives while keeping the same intent.

    Args:
        code: Python code string to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        tokens = [
            t for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type in (tokenize.NAME, tokenize.OP)
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = None  # malformed input -- let ast.parse below produce the error

    if tokens is not None:
        for i, tok in enumerate(tokens):
            if tok.type != tokenize.NAME:
                continue
            name = tok.string
            prev = tokens[i - 1].string if i > 0 else ""
            nxt = tokens[i + 1].string if i + 1 < len(tokens) else ""
            if prev == ".":
                continue  # an attribute access, not a bare name -- allowed
            if name in _FORBIDDEN_NAMES:
                return False, f"Potentially dangerous name detected: {name}"
            if name in _FORBIDDEN_BARE_CALLS and nxt == "(":
                return False, f"Potentially dangerous call detected: {name}()"
            if name in _FORBIDDEN_ATTRS and nxt == ".":
                attr = tokens[i + 2].string if i + 2 < len(tokens) else ""
                if attr in _FORBIDDEN_ATTRS[name]:
                    return False, f"Potentially dangerous call detected: {name}.{attr}"

    # Try to parse as valid Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, None


# Depth cap for _json_safe: guards against a pathological nested structure
# recursing without bound. Anything past this depth is repr()'d as-is rather
# than walked further.
_JSON_SAFE_MAX_DEPTH = 6


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Coerce a sandboxed ``result`` into something JSON-serializable.

    User code can assign anything to ``result`` -- a numpy scalar, an
    ndarray, an RDKit ``Mol``, a biotite ``AtomArray``, a set. Those reach
    ``ExecuteResponse.result`` in ``main.py`` unsanitized, and FastAPI/
    Pydantic only knows how to serialize JSON-native types, so a value like
    ``numpy.int64`` crashed the *response* with an unhandled 500 -- after the
    code had already executed successfully. Sanitizing once here, at the
    point ``result`` is captured, means every caller of ``execute_code``
    (the HTTP layer and direct callers such as tests) gets a safe value.

    Duck-typed rather than ``import numpy`` + ``isinstance``: a 0-d
    array-like with ``.item()``/``.dtype``/``.shape == ()`` is a scalar; an
    array-like with ``.tolist()``/``.shape`` is an array. This covers numpy
    (the sandbox's numeric library) without adding a hard dependency edge,
    and incidentally degrades gracefully for most other array libraries too.
    """
    # Exact type, not isinstance: numpy.float64 is a genuine subclass of
    # Python float (though numpy.int64/bool_ are not), so isinstance would
    # let a numpy.float64 pass through unconverted here and it would still
    # fail Pydantic's serializer downstream. Only real, plain Python scalars
    # short-circuit; everything else -- including numpy.float64 -- falls
    # through to the scalar-unwrap branch below.
    if value is None or type(value) in (bool, int, float, str):
        return value

    if _depth >= _JSON_SAFE_MAX_DEPTH:
        try:
            return repr(value)
        except Exception:
            return f"<unrepresentable {type(value).__name__}>"

    # 0-d array-like (numpy scalar, e.g. numpy.int64/float64/bool_).
    if (
        hasattr(value, "item")
        and callable(value.item)
        and hasattr(value, "dtype")
        and getattr(value, "shape", None) == ()
    ):
        try:
            return _json_safe(value.item(), _depth + 1)
        except Exception:
            pass

    # Array-like (numpy ndarray and similar).
    if hasattr(value, "tolist") and callable(value.tolist) and hasattr(value, "shape"):
        try:
            return _json_safe(value.tolist(), _depth + 1)
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth + 1) for v in value]

    # Last resort: an opaque object (RDKit Mol, Bio.PDB Structure, a custom
    # class) becomes its string form rather than crashing the response, so
    # the call still succeeds and the model sees something usable.
    try:
        return str(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


def execute_code(
    code: str,
    context: dict[str, Any] | None = None,
    timeout_seconds: int = 10,
    sandbox_context: SandboxContext | None = None,
) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment.
    
    Args:
        code: Python code to execute.
        context: Variables to inject into the execution context.
        timeout_seconds: Maximum execution time in seconds.
        sandbox_context: Optional context for data store access (aevom client).
        
    Returns:
        Dict with 'success', 'output', 'error', and 'result' fields.
    """
    # Sanitize code to remove markdown fencing that LLMs sometimes include
    code = sanitize_code(code)
    
    # Validate code first
    is_valid, error_msg = validate_code(code)
    if not is_valid:
        return {
            "success": False,
            "output": "",
            "error": error_msg,
            "result": None,
        }
    
    # Compile with RestrictedPython using our custom transformer
    # The ScientificComputingTransformer relaxes restrictions that are overly
    # burdensome for scientific computing (+=, nonlocal, *args/**kwargs)
    # while maintaining security-critical guards
    try:
        byte_code = compile_restricted(
            code,
            filename="<sandbox>",
            mode="exec",
            policy=ScientificComputingTransformer,
        )
        
        # byte_code is a Python code object if compilation succeeded
        if byte_code is None:
            return {
                "success": False,
                "output": "",
                "error": "Compilation failed: RestrictedPython returned None",
                "result": None,
            }
    except SyntaxError as e:
        return {
            "success": False,
            "output": "",
            "error": f"Syntax error: {e}",
            "result": None,
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"Compilation failed: {e}",
            "result": None,
        }
    
    # Create sandbox environment with optional aevom client.
    #
    # One namespace, not two. Passing separate globals and locals makes the
    # module body behave like a class body: top-level assignments land in
    # locals, but comprehensions, lambdas and def bodies open a new scope that
    # skips it and resolves against globals only. Snippets like
    #
    #     pat = Chem.MolFromSmarts("c1ccccc1")
    #     hits = [m for m in mols if m.HasSubstructMatch(pat)]
    #
    # then die on "NameError: name 'pat' is not defined", which is baffling to
    # anyone writing ordinary Python. Sharing one mapping restores module
    # semantics.
    namespace = create_sandbox_globals(context, sandbox_context)
    
    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Set up timeout (Unix only)
    old_handler = None
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # byte_code is a code object, execute it directly
            exec(byte_code, namespace)
        
        # Get the last expression result if any. Sanitized so a numpy scalar,
        # ndarray, or other non-JSON-native object the sandbox's libraries can
        # produce doesn't crash response serialization -- see _json_safe.
        result = _json_safe(namespace.get("result", namespace.get("_", None)))

        # Capture output from PrintCollector (RestrictedPython's print handling)
        # The _print variable in locals contains the PrintCollector instance
        # Calling it returns the accumulated print output
        print_collector = namespace.get("_print")
        if print_collector and callable(print_collector):
            printed_output = print_collector()
        else:
            printed_output = ""
        
        # Combine PrintCollector output with any direct stdout
        stdout_output = stdout_capture.getvalue()
        if printed_output and stdout_output:
            output = printed_output + stdout_output
        else:
            output = printed_output or stdout_output
        
        return {
            "success": True,
            "output": output,
            "error": stderr_capture.getvalue() or None,
            "result": result,
        }
        
    except TimeoutError:
        # Get any printed output before timeout
        print_collector = namespace.get("_print")
        printed_output = print_collector() if print_collector and callable(print_collector) else ""
        return {
            "success": False,
            "output": printed_output or stdout_capture.getvalue(),
            "error": f"Execution timed out after {timeout_seconds} seconds",
            "result": None,
        }
    except SandboxError as e:
        print_collector = namespace.get("_print")
        printed_output = print_collector() if print_collector and callable(print_collector) else ""
        return {
            "success": False,
            "output": printed_output or stdout_capture.getvalue(),
            "error": f"Security violation: {e}",
            "result": None,
        }
    except Exception as e:
        print_collector = namespace.get("_print")
        printed_output = print_collector() if print_collector and callable(print_collector) else ""
        return {
            "success": False,
            "output": printed_output or stdout_capture.getvalue(),
            "error": f"Execution error: {type(e).__name__}: {e}",
            "result": None,
        }
    finally:
        # Reset timeout
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler:
                signal.signal(signal.SIGALRM, old_handler)



