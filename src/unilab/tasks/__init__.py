"""Production task registry bootstrap.

Concrete task implementations live in this package.  The registry imports the
explicit leaf-module list directly, so registration stays deterministic and
does not depend on package discovery or import order.
"""

__unilab_registry_modules__ = (
    "unilab.tasks.locomotion.go2",
    "unilab.tasks.locomotion.g1",
    "unilab.tasks.navigation",
    "unilab.tasks.motion_tracking.g1",
)

__all__ = ["__unilab_registry_modules__"]
