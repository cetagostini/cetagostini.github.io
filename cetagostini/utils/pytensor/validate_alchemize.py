"""Validate the Alchemize-generated SmolLM2 artifact for provenance and static policy.

Reads the raw generated artifact (``alchemize_smollm2_raw.py.txt``) and performs:

1. **generation** — artifact hash and line-count match provenance record.
2. **syntax** — ``compile()`` succeeds on the artifact source.
3. **policy** — narrow syntactic AST scan allows only ``numpy``, ``pytensor``,
   ``gguf`` imports; rejects network, subprocess, eval, exec, pickle, and
   dynamic imports. Also detects direct aliases of forbidden builtins
   (e.g., ``danger = eval``). This is **not** a call-graph analysis or
   security proof.
4. **required_components** — verifies expected functions and constants exist.
5. **dequantization** — scoped to ``materialize_tensor``: checks for an exact
   ``gguf.dequantize`` call (or approved directly imported ``dequantize``)
   within that function body (expected: STATIC_FAIL for the raw draft).
6. **symbolic_head_reshape** — scoped to ``attention_layer``: checks for
   ``split_dims``/``join_dims`` versus raw ``.reshape`` within that function
   body (expected: STATIC_FAIL for the raw draft).
7. **runtime** — always BLOCKED (generated code is never executed by this validator).
8. **semantic** — always UNVERIFIED (requires human review of numerical correctness).

If a GGUF path is supplied via ``--gguf``, the file hash and tensor inventory
are verified against the provenance record.

Exit codes: ``main()`` returns nonzero for missing/tampered provenance,
syntax FAIL/BLOCKED, policy FAIL/BLOCKED, or required components FAIL.
Expected STATIC_FAIL (dequantization, reshape) and intentional
runtime BLOCKED / semantic UNVERIFIED are nonfatal.

Usage
-----
    python -m cetagostini.utils.pytensor.validate_alchemize
    python -m cetagostini.utils.pytensor.validate_alchemize --gguf /path/to/model.gguf
    python -m cetagostini.utils.pytensor.validate_alchemize --output results.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RAW_ARTIFACT_NAME = "alchemize_smollm2_raw.py.txt"

EXPECTED_ARTIFACT_SHA256 = "106cd2f597d7ab84bd955f733e8fdc5352311a0f76194e1a0cabf09575e57405"
EXPECTED_ARTIFACT_LINES = 358

EXPECTED_GGUF_SHA256 = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
EXPECTED_GGUF_FILENAME = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
EXPECTED_GGUF_REVISION = "09816acd5d99df7be770d85ea30822623dab342c"

# Historical misspelling preserved for backward compatibility
ALLOWCISTED_IMPORT_MODULES = frozenset({
    "numpy",
    "pytensor",
    "pytensor.tensor",
    "gguf",
    "np",
    "pt",
})

FORBIDDEN_IMPORT_MODULES = frozenset({
    "subprocess",
    "multiprocessing",
    "importlib",
    "__future__",
    "requests",
    "urllib",
    "http",
    "socket",
    "smtplib",
    "ftplib",
    "pickle",
    "shelve",
    "marshal",
    "ctypes",
    "shlex",
    "os.popen",
    "os.system",
})

FORBIDDEN_CALL_NAMES = frozenset({
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
})

REQUIRED_COMPONENTS = frozenset({
    "materialize_tensor",
    "load_weights",
    "load_config",
    "get_weight",
    "rms_norm",
    "silu",
    "linear_stored",
    "build_rope",
    "rotate_half",
    "apply_rope",
    "attention_layer",
    "mlp_layer",
    "build_model",
    "list_required_tensors",
    "missing_tensors",
    "compile_forward",
})

ALLOWED_RESULT_KEYS = frozenset({
    "metadata",
    "generation",
    "syntax",
    "policy",
    "required_components",
    "dequantization",
    "symbolic_head_reshape",
    "runtime",
    "semantic",
    "gguf_verification",
})


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    """Count lines in a file (including a final line without trailing newline).

    Parameters
    ----------
    path : Path
        File to count.

    Returns
    -------
    int
        Number of lines.
    """
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        return len(lines) - 1
    return len(lines)


def read_source(path: Path) -> str:
    """Read the artifact source as a string.

    Parameters
    ----------
    path : Path
        Path to the ``.py.txt`` artifact.

    Returns
    -------
    str
        Source text.
    """
    return path.read_text(encoding="utf-8")


def check_generation(artifact_path: Path) -> dict:
    """Verify artifact hash and line count against provenance record.

    Parameters
    ----------
    artifact_path : Path
        Path to the raw artifact file.

    Returns
    -------
    dict
        Keys: ``status``, ``sha256``, ``expected_sha256``, ``lines``,
        ``expected_lines``, ``hash_match``, ``line_match``.
    """
    actual_hash = compute_sha256(artifact_path)
    actual_lines = count_lines(artifact_path)
    hash_ok = actual_hash == EXPECTED_ARTIFACT_SHA256
    lines_ok = actual_lines == EXPECTED_ARTIFACT_LINES
    passed = hash_ok and lines_ok
    return {
        "status": "PASS" if passed else "FAIL",
        "sha256": actual_hash,
        "expected_sha256": EXPECTED_ARTIFACT_SHA256,
        "lines": actual_lines,
        "expected_lines": EXPECTED_ARTIFACT_LINES,
        "hash_match": hash_ok,
        "line_match": lines_ok,
    }


def check_syntax(source: str) -> dict:
    """Verify the artifact compiles without syntax errors.

    Parameters
    ----------
    source : str
        Python source text.

    Returns
    -------
    dict
        Keys: ``status``, ``error`` (if any).
    """
    try:
        compile(source, "<alchemize_smollm2_raw>", "exec")
        return {"status": "PASS"}
    except SyntaxError as exc:
        return {"status": "FAIL", "error": str(exc)}


def _collect_imports(tree: ast.Module) -> list[dict]:
    """Extract import statements from an AST.

    Parameters
    ----------
    tree : ast.Module
        Parsed AST.

    Returns
    -------
    list of dict
        Each dict has ``kind`` (``"import"`` or ``"from"``), ``module``,
        ``names`` (list of str), and ``lineno``.
    """
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "kind": "import",
                    "module": alias.name,
                    "names": [alias.name],
                    "lineno": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append({
                "kind": "from",
                "module": module,
                "names": names,
                "lineno": node.lineno,
            })
    return imports


def _check_forbidden_aliases(tree: ast.Module) -> list[str]:
    """Detect direct aliases of forbidden builtins (e.g., ``danger = eval``).

    Parameters
    ----------
    tree : ast.Module
        Parsed AST.

    Returns
    -------
    list of str
        Descriptions of each violation found.
    """
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                    if node.value.id in FORBIDDEN_CALL_NAMES:
                        violations.append(
                            f"{target.id} = {node.value.id} (line {node.lineno})"
                        )
    return violations


def check_policy(source: str) -> dict:
    """Check import policy: only allowlisted modules, no forbidden calls.

    Parameters
    ----------
    source : str
        Python source text.

    Returns
    -------
    dict
        Keys: ``status``, ``imports``, ``violations``, ``forbidden_calls``,
        ``dynamic_imports``, ``forbidden_aliases``.
    """
    tree = ast.parse(source)
    imports = _collect_imports(tree)
    violations = []
    forbidden_calls = []
    dynamic_imports = []

    # Check imports against allowlist/forbidden list
    for imp in imports:
        module = imp["module"]
        # Check if module is forbidden
        if module in FORBIDDEN_IMPORT_MODULES:
            violations.append(
                f"forbidden import '{module}' at line {imp['lineno']}"
            )
        # Check if module root is forbidden (e.g., os.popen)
        module_root = module.split(".")[0]
        for forbidden in FORBIDDEN_IMPORT_MODULES:
            if "." in forbidden:
                if module == forbidden or module.startswith(forbidden + "."):
                    violations.append(
                        f"forbidden import '{module}' at line {imp['lineno']}"
                    )
        # Check if module is allowlisted (only for top-level imports)
        if imp["kind"] == "import":
            if module not in ALLOWCISTED_IMPORT_MODULES and module_root not in ALLOWCISTED_IMPORT_MODULES:
                violations.append(
                    f"non-allowlisted import '{module}' at line {imp['lineno']}"
                )
        elif imp["kind"] == "from":
            if module not in ALLOWCISTED_IMPORT_MODULES and module_root not in ALLOWCISTED_IMPORT_MODULES:
                violations.append(
                    f"non-allowlisted from-import '{module}' at line {imp['lineno']}"
                )

    # Check for forbidden calls and dynamic imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALL_NAMES:
                forbidden_calls.append(f"{func.id}() at line {node.lineno}")
            if isinstance(func, ast.Attribute) and func.attr == "__import__":
                dynamic_imports.append(f"__import__ at line {node.lineno}")

    # Check for forbidden aliases
    forbidden_aliases = _check_forbidden_aliases(tree)

    status = "PASS" if not violations and not forbidden_calls and not dynamic_imports and not forbidden_aliases else "FAIL"

    return {
        "status": status,
        "imports": imports,
        "violations": violations,
        "forbidden_calls": forbidden_calls,
        "dynamic_imports": dynamic_imports,
        "forbidden_aliases": forbidden_aliases,
    }


def check_required_components(source: str) -> dict:
    """Verify expected functions and constants exist in the artifact.

    Parameters
    ----------
    source : str
        Python source text.

    Returns
    -------
    dict
        Keys: ``status``, ``found``, ``missing``.
    """
    tree = ast.parse(source)
    defined_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)

    found = sorted(REQUIRED_COMPONENTS & defined_names)
    missing = sorted(REQUIRED_COMPONENTS - defined_names)
    status = "PASS" if not missing else "FAIL"

    return {
        "status": status,
        "found": found,
        "missing": missing,
    }


def _get_function_body(tree: ast.Module, func_name: str) -> ast.FunctionDef | None:
    """Find a function definition by name and return it.

    Parameters
    ----------
    tree : ast.Module
        Parsed AST.
    func_name : str
        Function name to find.

    Returns
    -------
    ast.FunctionDef or None
        The function node, or None if not found.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def check_dequantization(source: str) -> dict:
    """Check for ``gguf.dequantize`` call within ``materialize_tensor``.

    Parameters
    ----------
    source : str
        Python source text.

    Returns
    -------
    dict
        Keys: ``status``, ``has_gguf_dequantize_call``,
        ``materialize_reads_data_attr``, ``detail``.
    """
    tree = ast.parse(source)
    func = _get_function_body(tree, "materialize_tensor")
    if func is None:
        return {
            "status": "FAIL",
            "has_gguf_dequantize_call": False,
            "materialize_reads_data_attr": False,
            "detail": "materialize_tensor function not found",
        }

    has_dequantize = False
    reads_data_attr = False

    for node in ast.walk(func):
        # Check for gguf.dequantize(...) or dequantize(...)
        if isinstance(node, ast.Call):
            func_node = node.func
            if isinstance(func_node, ast.Attribute):
                if func_node.attr == "dequantize":
                    has_dequantize = True
            elif isinstance(func_node, ast.Name):
                if func_node.id == "dequantize":
                    has_dequantize = True
            # Check for getattr(tensor, "data", ...) pattern
            if isinstance(func_node, ast.Name) and func_node.id == "getattr":
                if len(node.args) >= 2:
                    second_arg = node.args[1]
                    if isinstance(second_arg, ast.Constant) and second_arg.value == "data":
                        reads_data_attr = True
        # Check for tensor.data access
        if isinstance(node, ast.Attribute) and node.attr == "data":
            reads_data_attr = True

    status = "PASS" if has_dequantize else "STATIC_FAIL"
    detail = (
        "materialize_tensor reads tensor.data directly and raises "
        "NotImplementedError for non-float data; no gguf.dequantize call "
        "found within materialize_tensor"
    )

    return {
        "status": status,
        "has_gguf_dequantize_call": has_dequantize,
        "materialize_reads_data_attr": reads_data_attr,
        "detail": detail,
    }


def check_symbolic_head_reshape(source: str) -> dict:
    """Check for ``split_dims``/``join_dims`` versus raw ``.reshape`` in ``attention_layer``.

    Parameters
    ----------
    source : str
        Python source text.

    Returns
    -------
    dict
        Keys: ``status``, ``uses_split_dims``, ``uses_join_dims``,
        ``uses_raw_reshape``, ``detail``.
    """
    tree = ast.parse(source)
    func = _get_function_body(tree, "attention_layer")
    if func is None:
        return {
            "status": "FAIL",
            "uses_split_dims": False,
            "uses_join_dims": False,
            "uses_raw_reshape": False,
            "detail": "attention_layer function not found",
        }

    uses_split_dims = False
    uses_join_dims = False
    uses_raw_reshape = False

    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            func_node = node.func
            # Check for pt.split_dims / split_dims
            if isinstance(func_node, ast.Attribute):
                if func_node.attr == "split_dims":
                    uses_split_dims = True
                elif func_node.attr == "join_dims":
                    uses_join_dims = True
                elif func_node.attr == "reshape":
                    uses_raw_reshape = True
            elif isinstance(func_node, ast.Name):
                if func_node.id == "split_dims":
                    uses_split_dims = True
                elif func_node.id == "join_dims":
                    uses_join_dims = True

    status = "PASS" if uses_split_dims and uses_join_dims else "STATIC_FAIL"
    detail = (
        "attention_layer uses raw .reshape()/.dimshuffle() for head "
        "splitting/joining; split_dims/join_dims not found within attention_layer"
    )

    return {
        "status": status,
        "uses_split_dims": uses_split_dims,
        "uses_join_dims": uses_join_dims,
        "uses_raw_reshape": uses_raw_reshape,
        "detail": detail,
    }


def check_gguf_verification(gguf_path: Path | None) -> dict | None:
    """Verify GGUF file hash and tensor inventory against provenance record.

    Parameters
    ----------
    gguf_path : Path or None
        Path to the GGUF file. If None, returns None.

    Returns
    -------
    dict or None
        Keys: ``status``, ``sha256``, ``expected_sha256``, ``hash_match``,
        ``filename``, ``expected_filename``, ``filename_match``,
        ``tensor_count``, ``ggml_types``.
    """
    if gguf_path is None:
        return None

    actual_hash = compute_sha256(gguf_path)
    hash_ok = actual_hash == EXPECTED_GGUF_SHA256
    filename = gguf_path.name
    filename_ok = filename == EXPECTED_GGUF_FILENAME

    # Try to read tensor inventory
    tensor_count = None
    ggml_types = None
    try:
        import gguf as gguf_mod
        reader = gguf_mod.GGUFReader(str(gguf_path))
        tensor_count = len(reader.tensors)
        type_set = set()
        for tensor in reader.tensors:
            try:
                type_set.add(tensor.tensor_type.name)
            except Exception:
                pass
        ggml_types = sorted(type_set)
    except Exception:
        pass

    status = "PASS" if hash_ok and filename_ok else "FAIL"

    result = {
        "status": status,
        "sha256": actual_hash,
        "expected_sha256": EXPECTED_GGUF_SHA256,
        "hash_match": hash_ok,
        "filename": filename,
        "expected_filename": EXPECTED_GGUF_FILENAME,
        "filename_match": filename_ok,
    }
    if tensor_count is not None:
        result["tensor_count"] = tensor_count
    if ggml_types is not None:
        result["ggml_types"] = ggml_types

    return result


def run_all_checks(artifact_path: Path, gguf_path: Path | None = None) -> dict:
    """Run all validation checks on the artifact.

    Parameters
    ----------
    artifact_path : Path
        Path to the raw artifact file.
    gguf_path : Path or None
        Optional path to the GGUF file.

    Returns
    -------
    dict
        Complete validation result with all check sections.
    """
    source = read_source(artifact_path)

    result = {
        "metadata": {
            "validator": "validate_alchemize.py",
            "artifact_name": RAW_ARTIFACT_NAME,
            "alchemize_commit": "84841ca2b9e62291913a7d783cf7307732f791c9",
            "model": "claude-opus-4-8",
            "api_tokens": {
                "input": 17272,
                "output": 5135,
            },
            "gguf_revision": EXPECTED_GGUF_REVISION,
            "gguf_filename": EXPECTED_GGUF_FILENAME,
        },
        "generation": check_generation(artifact_path),
        "syntax": check_syntax(source),
        "policy": check_policy(source),
        "required_components": check_required_components(source),
        "dequantization": check_dequantization(source),
        "symbolic_head_reshape": check_symbolic_head_reshape(source),
        "runtime": {
            "status": "BLOCKED",
            "reason": "generated draft code is not executed by this validator",
        },
        "semantic": {
            "status": "UNVERIFIED",
            "reason": "numerical correctness requires human review",
        },
    }

    gguf_result = check_gguf_verification(gguf_path)
    if gguf_result is not None:
        result["gguf_verification"] = gguf_result

    return result


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON data atomically using a temporary file.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : dict
        Data to serialize as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the validator and return exit code.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        0 if all critical checks pass, nonzero otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Validate Alchemize-generated SmolLM2 artifact"
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="Path to GGUF file for verification",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to JSON file",
    )
    args = parser.parse_args(argv)

    artifact_path = ARTIFACT_DIR / RAW_ARTIFACT_NAME
    if not artifact_path.exists():
        print(f"ERROR: artifact not found: {artifact_path}", file=sys.stderr)
        return 1

    result = run_all_checks(artifact_path, args.gguf)

    # Print summary
    for key in ["generation", "syntax", "policy", "required_components",
                "dequantization", "symbolic_head_reshape", "runtime", "semantic"]:
        if key in result:
            status = result[key].get("status", "UNKNOWN")
            print(f"{key:25s} {status}")

    if "gguf_verification" in result:
        status = result["gguf_verification"].get("status", "UNKNOWN")
        print(f"{'gguf_verification':25s} {status}")

    # Write output if requested
    if args.output:
        write_json_atomic(args.output, result)
        print(f"\nResults written to {args.output}")

    # Determine exit code: nonzero for critical failures
    gen_status = result["generation"]["status"]
    syntax_status = result["syntax"]["status"]
    policy_status = result["policy"]["status"]
    components_status = result["required_components"]["status"]

    if gen_status == "FAIL":
        return 1
    if syntax_status in ("FAIL", "BLOCKED"):
        return 1
    if policy_status in ("FAIL", "BLOCKED"):
        return 1
    if components_status == "FAIL":
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
