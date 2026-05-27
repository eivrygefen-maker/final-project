#!/usr/bin/env python3
"""No-EPS SLEPc API availability probe via setType/getType (no eps.solve)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _eps_settype_probe(eps: Any, name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return (available, getType_after_set, error)."""
    try:
        eps.setType(name)
        got = str(eps.getType()).strip().lower()
        return True, got, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def slepc_eps_api_probe() -> Dict[str, Any]:
    """
    Probe VM/runtime SLEPc without calling eps.solve().

    Uses EPS().create(); eps.setType(name); eps.getType() for each solver type.
    """
    out: Dict[str, Any] = {
        "vm_slepc_import_pass": False,
        "petsc_version": None,
        "vm_slepc_version": None,
        "jd_api_available": False,
        "gd_api_available": False,
        "ciss_api_available": False,
        "krylovschur_api_available": False,
        "ciss_region_api_available": False,
        "new_dependency_required": True,
        "recommended_primary_solver_api_status": "UNKNOWN_IMPORT_FAILED",
        "eps_settype_probe": {},
        "eps_type_enum_exposure": {},
        "rg_probe": {},
        "no_eps_solve_called": True,
    }
    try:
        from petsc4py import PETSc
        import SLEPc

        out["vm_slepc_import_pass"] = True
        out["petsc_version"] = str(PETSc.Sys.getVersion())
        out["vm_slepc_version"] = str(getattr(SLEPc, "__version__", "unknown"))
        out["new_dependency_required"] = False

        for enum_name in ("KRYLOVSCHUR", "CISS", "JD", "GD"):
            t = getattr(SLEPc.EPS.Type, enum_name, None)
            out["eps_type_enum_exposure"][enum_name] = t is not None

        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        probes: Dict[str, Any] = {}
        for key, st_name in (
            ("krylovschur", "krylovschur"),
            ("ciss", "ciss"),
            ("jd", "jd"),
            ("gd", "gd"),
        ):
            ok, got, err = _eps_settype_probe(eps, st_name)
            probes[key] = {
                "setType_argument": st_name,
                "available": ok,
                "getType_after_set": got,
                "error": err,
            }
        out["eps_settype_probe"] = probes
        out["krylovschur_api_available"] = bool(probes.get("krylovschur", {}).get("available"))
        out["ciss_api_available"] = bool(probes.get("ciss", {}).get("available"))
        out["jd_api_available"] = bool(probes.get("jd", {}).get("available"))
        out["gd_api_available"] = bool(probes.get("gd", {}).get("available"))

        rg_types: List[str] = []
        rg_create_ok = False
        rg_interval_ok = False
        if hasattr(SLEPc, "RG") and hasattr(SLEPc.RG, "Type"):
            for rg_name in ("INTERVAL", "ELLIPSE", "POLYGON", "RING"):
                if getattr(SLEPc.RG.Type, rg_name, None) is not None:
                    rg_types.append(rg_name.lower())
            rg_interval_ok = getattr(SLEPc.RG.Type, "INTERVAL", None) is not None
        try:
            rg = SLEPc.RG().create(PETSc.COMM_WORLD)
            rg_create_ok = True
            rg.destroy()
        except Exception as exc:
            out["rg_probe"]["create_error"] = f"{type(exc).__name__}: {exc}"
        out["rg_probe"] = {
            "rg_create_available": rg_create_ok,
            "rg_types_enum": rg_types,
            "rg_interval_enum": rg_interval_ok,
            "eps_has_setRG": hasattr(eps, "setRG"),
        }
        out["ciss_region_api_available"] = bool(
            rg_create_ok and rg_interval_ok and out["rg_probe"].get("eps_has_setRG")
        )

        try:
            eps.destroy()
        except Exception:
            pass

        if out["jd_api_available"] and out["gd_api_available"]:
            out["recommended_primary_solver_api_status"] = (
                "AVAILABLE_REQUIRES_CLEANED_FORMULATION_AND_DISPATCH_INTEGRATION"
            )
        elif out["jd_api_available"] or out["gd_api_available"]:
            out["recommended_primary_solver_api_status"] = (
                "PARTIAL_JD_GD_AVAILABLE_REQUIRES_DISPATCH_REVIEW"
            )
        else:
            out["recommended_primary_solver_api_status"] = "JD_GD_NOT_AVAILABLE_ON_VM"

    except Exception as exc:
        out["import_error"] = f"{type(exc).__name__}: {exc}"

    return out
