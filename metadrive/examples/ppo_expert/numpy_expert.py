"""Placeholder for the optional MetaDrive PPO expert example.

The original MetaDrive distribution includes a small PPO expert weight file for
interactive demo policies. SAGE does not use that policy, so the file is not
bundled in this release.
"""


def expert(*_args, **_kwargs):
    raise ValueError(
        "The optional MetaDrive PPO expert weights are not included in the "
        "SAGE release. Use the SAGE training and evaluation scripts instead."
    )
