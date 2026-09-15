from pathlib import Path
import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


# Repository root, regardless of where this script is executed from.
ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "schemas"


SCHEMA_FILES = {
    "historical_snapshot": "historical_snapshot.schema.json",
    "scout_output": "scout_output.schema.json",
    "research_scout_output": "research_scout_output.schema.json",
    "research_universe": "research_universe.schema.json",
    "research_alpha_batch": "research_alpha_batch.schema.json",
    "end_of_day_outcome": "end_of_day_outcome.schema.json",
    "benchmark_result": "benchmark_result.schema.json",
    "postmortem": "postmortem.schema.json",
    "feature_proposal": "feature_proposal.schema.json",
    "validation_result": "validation_result.schema.json",
    "promotion_decision": "promotion_decision.schema.json",
}


class ContractError(Exception):
    """Raised when a Scout Trainer contract cannot be validated."""


def load_json(path: Path) -> dict:
    """Load and return a JSON file."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ContractError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc


def load_schema(contract_name: str) -> dict:
    """Load a contract schema by its registered name."""
    if contract_name not in SCHEMA_FILES:
        raise ContractError(f"Unknown contract: {contract_name}")

    return load_json(SCHEMA_DIR / SCHEMA_FILES[contract_name])


def validate_schema_definition(contract_name: str) -> None:
    """
    Validate that one of our schema files is itself a valid
    JSON Schema Draft 2020-12 definition.
    """
    schema = load_schema(contract_name)

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ContractError(
            f"Invalid schema definition for {contract_name}: {exc.message}"
        ) from exc


def validate_contract(contract_name: str, payload: dict) -> None:
    """
    Validate a payload against one of the Scout Trainer contracts.

    Raises ContractError if validation fails.
    """
    schema = load_schema(contract_name)

    try:
        validator = Draft202012Validator(
            schema,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )
        validator.validate(payload)

    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        location = location or "<root>"

        raise ContractError(
            f"{contract_name} validation failed at "
            f"{location}: {exc.message}"
        ) from exc


def validate_all_schema_definitions() -> None:
    """Validate every registered schema definition."""
    failures = []

    for contract_name in SCHEMA_FILES:
        try:
            validate_schema_definition(contract_name)
            print(f"PASS  {contract_name}")
        except ContractError as exc:
            failures.append(str(exc))
            print(f"FAIL  {contract_name}")

    if failures:
        print("\nSchema validation failures:")
        for failure in failures:
            print(f" - {failure}")

        raise SystemExit(1)

    print(
        f"\nAll {len(SCHEMA_FILES)} Scout Trainer "
        "contract schemas are valid."
    )


if __name__ == "__main__":
    validate_all_schema_definitions()
