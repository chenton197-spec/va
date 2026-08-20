# Repository Guidelines

## Framework Boundaries

`teleop_sdk` is a device-agnostic teleoperation framework. Keep control flow
in `controller.py`, public contracts in `interfaces.py`, reusable math in
`algorithms/` and `filters.py`, and hardware implementations in `adapters/`.
The programs under `examples/` compose the bundled Alicia-D, FR3, and
Gloria-M reference adapters; they must not define dependencies for the core.
`teleop.yaml` configures that reference deployment, not the generic controller.

Do not import a vendor SDK, use a vendor method name, or add device-specific
configuration to the controller or shared algorithms. New hardware belongs in
an adapter and should follow `ADAPTER_GUIDE.md`.

## Data and Lifecycle Contracts

Public joint arrays are `numpy.ndarray` values in degrees. Adapters convert
vendor units at their boundary. Gripper values are normalized from `0.0`
(closed) to `1.0` (open). `LeaderArm.joint_count` and
`FollowerArm.joint_count` are fixed adapter capabilities and must agree.
Followers supply default joint limits and own connection, servo start/stop,
recovery, and cleanup. Keep vendor exceptions and resource handling there.

## Development and Testing

Install the checked-in dependencies plus NumPy, then run the hardware-free
suite from the repository root:

```bash
python -m pip install -r requirements.txt numpy
python -m unittest discover -s teleop_sdk/tests -p 'test_*.py' -v
```

Use `examples/test_no_bot.py` or `MockFollower` before exercising real
hardware. Do not make automated tests require a robot, serial port, GUI, or
vendor SDK.

## Style and Tests

Use four-space indentation, type annotations, and the existing `snake_case`
and `PascalCase` naming. Preserve the controller's generic behavior rather
than special-casing a shipped adapter. Place tests in `teleop_sdk/tests/`, name
them `test_*.py`, and use `unittest` with fakes or mocks. Add direct coverage
for changes to mappings, limits, lifecycle, unit conversion, or recovery.

## Changes and Reviews

This repository has no commit history yet; use concise imperative Conventional
Commit-style subjects, such as `feat: add kinova follower adapter`. In a pull
request, state the affected contracts, configuration changes, hardware
assumptions, and test command run. Never commit credentials, production device
addresses, or local vendor SDK files.
