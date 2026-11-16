#!/usr/bin/env python3
"""A self-contained, 500-line expense tracker sample.

Standard-library-only Python that demonstrates value objects, Decimal money,
a JSON-backed repository with transactions, aggregation helpers and an
``argparse`` CLI. It is meant to be read and extended, not used in production.

Try it from the repository root::

    python3 examples/expense_tracker.py demo
    python3 examples/expense_tracker.py --db /tmp/demo.json add 12.50 \\
        --category food --note "Noodles"
    python3 examples/expense_tracker.py --db /tmp/demo.json report
    python3 -m doctest examples/expense_tracker.py

Design notes:
    Validation and storage failures use separate error types. Mutations can
    be grouped in a transaction, and writes replace the JSON document
    atomically so a crash cannot leave a half-written file behind.
    The CLI stays thin so the core logic remains easy to reuse and test.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

CENTS = Decimal("0.01")
CURRENCY = "CNY"


# --- Money, date and month helpers -----------------------------------------


def money(value: str | int | Decimal) -> Decimal:
    """Return *value* as a two-decimal ``Decimal`` amount (half-up).

    ``float`` is rejected so callers never inherit binary floating point
    surprises.

    >>> money("1.005")
    Decimal('1.01')
    >>> money(3)
    Decimal('3.00')
    >>> money("oops")
    Traceback (most recent call last):
        ...
    ValueError: not a valid amount: 'oops'
    >>> money(1.5)
    Traceback (most recent call last):
        ...
    TypeError: money() does not accept float
    """
    if isinstance(value, float):
        raise TypeError("money() does not accept float")
    try:
        return Decimal(str(value)).quantize(CENTS, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError(f"not a valid amount: {value!r}") from exc


def format_money(amount: Decimal) -> str:
    """Format an amount for reports.

    >>> format_money(money("-3"))
    '-CNY 3.00'
    """
    amount = money(amount)
    sign = "-" if amount < 0 else ""
    return f"{sign}{CURRENCY} {abs(amount):.2f}"


def parse_date(value: str) -> date:
    """Parse an ISO-8601 date such as ``2025-02-03``.

    >>> parse_date("2025-02-03")
    datetime.date(2025, 2, 3)
    """
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"expected an ISO date (YYYY-MM-DD), got {value!r}"
        ) from exc


def month_key(day: date) -> str:
    """Return the ``YYYY-MM`` bucket of *day*.

    >>> month_key(date(2025, 2, 3))
    '2025-02'
    """
    return f"{day.year:04d}-{day.month:02d}"


# --- Errors and value objects ----------------------------------------------


class ExpenseError(Exception):
    """Base class for every error raised by this module."""


class ValidationError(ExpenseError, ValueError):
    """Raised when an expense record is semantically invalid."""


class StoreError(ExpenseError, RuntimeError):
    """Raised when storage cannot be read or written."""


class Category(str, Enum):
    """Categories an expense may belong to.

    Subclassing ``str`` keeps values JSON-friendly and comparable with
    plain strings.

    >>> Category.parse("FOOD")
    <Category.FOOD: 'food'>
    >>> Category.parse("travel")
    Traceback (most recent call last):
        ...
    expense_tracker.ValidationError: unknown category: 'travel'; choose one of: food, transport, housing, utilities, health, entertainment, other
    """

    FOOD = "food"
    TRANSPORT = "transport"
    HOUSING = "housing"
    UTILITIES = "utilities"
    HEALTH = "health"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"

    @classmethod
    def parse(cls, value: str | Category) -> Category:
        """Build a category from a case-insensitive name or another category."""
        if isinstance(value, cls):
            return value
        name = str(value).strip().lower()
        if name not in {item.value for item in cls}:
            choices = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"unknown category: {value!r}; choose one of: {choices}"
            )
        return cls(name)


@dataclass(frozen=True, slots=True)
class Expense:
    """A single expense entry.

    >>> item = Expense(1, date(2025, 2, 3), Category.FOOD, "12.505", "Lunch")
    >>> item.amount, item.month
    (Decimal('12.51'), '2025-02')
    """

    expense_id: int
    day: date
    category: Category
    amount: Decimal
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.expense_id, int) or self.expense_id < 0:
            raise ValidationError("expense_id must be a non-negative integer")
        if not isinstance(self.day, date):
            raise ValidationError("day must be a datetime.date")
        object.__setattr__(self, "category", Category.parse(self.category))
        try:
            amount = money(self.amount)
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        if amount <= 0:
            raise ValidationError("amount must be greater than zero")
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "note", str(self.note).strip())

    @property
    def month(self) -> str:
        """The ``YYYY-MM`` bucket of this expense."""
        return month_key(self.day)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.expense_id,
            "day": self.day.isoformat(),
            "category": self.category.value,
            "amount": str(self.amount),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Expense:
        """Rebuild an expense from :meth:`as_dict` output."""
        try:
            return cls(
                expense_id=int(data["id"]),
                day=parse_date(str(data["day"])),
                category=Category.parse(str(data["category"])),
                amount=Decimal(str(data["amount"])),
                note=str(data.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"invalid expense record: {data!r}") from exc


# --- JSON-backed repository -------------------------------------------------


class ExpenseStore:
    """A tiny repository of expenses persisted as JSON.

    State lives in memory and the whole document is written on each
    mutation. :meth:`transaction` groups mutations into a single write.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._expenses: list[Expense] = []
        self._next_id = 1
        self._dirty = False
        self._in_transaction = False
        if self.path is not None and self.path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self._expenses)

    @contextmanager
    def transaction(self) -> Iterator[ExpenseStore]:
        """Batch mutations and persist them once when the block exits.

        >>> store = ExpenseStore()
        >>> with store.transaction():
        ...     _ = store.add("food", "10", date(2025, 2, 1))
        ...     _ = store.add("health", "20", date(2025, 2, 2))
        >>> len(store)
        2
        """
        previous = self._in_transaction
        self._in_transaction = True
        try:
            yield self
        finally:
            self._in_transaction = previous
            if not previous and self._dirty:
                self.save()

    def load(self) -> None:
        """Replace in-memory state with the JSON file contents."""
        if self.path is None:
            raise StoreError("cannot load a store without a path")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StoreError(f"store not found: {self.path}") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read store {self.path}: {exc}") from exc
        self._expenses = [
            Expense.from_dict(item) for item in raw.get("expenses", [])
        ]
        self._next_id = int(raw.get("next_id", 1))
        self._dirty = False

    def save(self) -> None:
        """Atomically write the current state; a no-op for memory stores."""
        if self.path is None:
            self._dirty = False
            return
        document = {
            "next_id": self._next_id,
            "expenses": [item.as_dict() for item in self._expenses],
        }
        payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        temp = self.path.with_name(self.path.name + ".tmp")
        try:
            temp.write_text(payload, encoding="utf-8")
            temp.replace(self.path)
        except OSError as exc:
            raise StoreError(f"cannot write store {self.path}: {exc}") from exc
        self._dirty = False

    def _persist(self) -> None:
        self._dirty = True
        if not self._in_transaction:
            self.save()

    def add(
        self,
        category: Category | str,
        amount: str | int | Decimal,
        day: date,
        note: str = "",
    ) -> Expense:
        """Create, store and return a new expense."""
        item = Expense(self._next_id, day, category, amount, note)  # type: ignore[arg-type]
        self._expenses.append(item)
        self._next_id += 1
        self._persist()
        return item

    def remove(self, expense_id: int) -> Expense:
        """Remove and return the expense with *expense_id*."""
        for index, item in enumerate(self._expenses):
            if item.expense_id == expense_id:
                del self._expenses[index]
                self._persist()
                return item
        raise KeyError(f"no expense with id {expense_id}")

    def expenses(
        self,
        month: str | None = None,
        category: Category | str | None = None,
    ) -> list[Expense]:
        """Return matching expenses ordered by day, then id."""
        wanted = Category.parse(category) if category is not None else None
        selected = [
            item
            for item in self._expenses
            if (month is None or item.month == month)
            and (wanted is None or item.category is wanted)
        ]
        return sorted(selected, key=lambda item: (item.day, item.expense_id))

    def total(self, month: str | None = None) -> Decimal:
        """Return the total amount spent, optionally limited to one month."""
        return sum(
            (item.amount for item in self.expenses(month=month)),
            start=Decimal("0.00"),
        )


# --- Aggregation helpers ----------------------------------------------------


def category_totals(expenses: Iterable[Expense]) -> dict[Category, Decimal]:
    """Sum expenses per category, in enum order.

    >>> items = [
    ...     Expense(1, date(2025, 2, 1), Category.FOOD, "5"),
    ...     Expense(2, date(2025, 2, 2), Category.FOOD, "7"),
    ...     Expense(3, date(2025, 2, 3), Category.HEALTH, "2"),
    ... ]
    >>> {key.value: str(value) for key, value in category_totals(items).items()}
    {'food': '12.00', 'health': '2.00'}
    """
    totals: dict[Category, Decimal] = {}
    for item in expenses:
        total = totals.get(item.category, Decimal("0.00"))
        totals[item.category] = total + item.amount
    return {category: totals[category] for category in Category if category in totals}


def monthly_totals(expenses: Iterable[Expense]) -> dict[str, Decimal]:
    """Sum expenses per ``YYYY-MM`` bucket, oldest first.

    >>> items = [
    ...     Expense(1, date(2025, 1, 5), Category.FOOD, "10"),
    ...     Expense(2, date(2025, 2, 5), Category.FOOD, "20"),
    ...     Expense(3, date(2025, 1, 6), Category.FOOD, "1"),
    ... ]
    >>> {key: str(value) for key, value in monthly_totals(items).items()}
    {'2025-01': '11.00', '2025-02': '20.00'}
    """
    totals: dict[str, Decimal] = {}
    for item in expenses:
        totals[item.month] = totals.get(item.month, Decimal("0.00")) + item.amount
    return dict(sorted(totals.items()))


def iter_largest(
    expenses: Iterable[Expense], count: int = 5
) -> Iterator[Expense]:
    """Yield the *count* largest expenses, largest first."""
    ordered = sorted(
        expenses, key=lambda item: (item.amount, item.expense_id), reverse=True
    )
    yield from ordered[:count]


def format_report(store: ExpenseStore, month: str | None = None) -> str:
    """Render a text report for *month* (or all months)."""
    expenses = store.expenses(month=month)
    lines = [
        f"Expense report for {month or 'all months'}",
        "=" * 60,
        f"Entries: {len(expenses)}",
        f"Total:   {format_money(store.total(month=month))}",
        "",
        "By category:",
    ]
    totals = category_totals(expenses)
    if not totals:
        lines.append("  (no expenses)")
    for category, amount in totals.items():
        lines.append(f"  {category.value:<14} {format_money(amount):>14}")
    lines += ["", "Largest expenses:"]
    largest = list(iter_largest(expenses))
    if not largest:
        lines.append("  (no expenses)")
    for item in largest:
        note = f"  # {item.note}" if item.note else ""
        lines.append(
            f"  {item.day.isoformat()}  {item.category.value:<14}"
            f" {format_money(item.amount):>14}{note}"
        )
    return "\n".join(lines)


# --- Command line interface -------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    categories = [item.value for item in Category]
    parser = argparse.ArgumentParser(
        prog="expense_tracker",
        description="Track expenses in a JSON file.",
    )
    parser.add_argument(
        "--db", default="expenses.json", help="path to the JSON document"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="add an expense")
    add.add_argument("amount", help="positive amount, e.g. 12.50")
    add.add_argument("--category", default=Category.OTHER.value, choices=categories)
    add.add_argument("--day", default=date.today().isoformat(), help="ISO date")
    add.add_argument("--note", default="", help="free-form note")

    listing = commands.add_parser("list", help="list expenses")
    listing.add_argument("--month", help="limit to YYYY-MM")
    listing.add_argument("--category", choices=categories)

    report = commands.add_parser("report", help="show a summary report")
    report.add_argument("--month", help="limit to YYYY-MM")

    commands.add_parser("demo", help="run a self-contained demonstration")
    return parser


def run_demo() -> int:
    """Exercise the module end to end and print the result."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "demo.json"
        store = ExpenseStore(path)
        with store.transaction():
            store.add(Category.FOOD, "38.50", date(2025, 2, 1), "Lunch")
            store.add(Category.TRANSPORT, "6", date(2025, 2, 1), "Metro")
            store.add(Category.FOOD, "52.80", date(2025, 2, 2), "Dinner")
            store.add(Category.HEALTH, "120", date(2025, 2, 3), "Pharmacy")
            store.add(Category.ENTERTAINMENT, "60", date(2025, 2, 4), "Cinema")
        print(format_report(store, month="2025-02"))
        print(f"\nPersisted document: {path}")
        print(f"Reloaded entries:   {len(ExpenseStore(path))}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse *argv* and dispatch to a command; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return run_demo()
    try:
        store = ExpenseStore(args.db)
        if args.command == "add":
            item = store.add(
                args.category, args.amount, parse_date(args.day), args.note
            )
            print(f"Added #{item.expense_id}: {format_money(item.amount)}")
        elif args.command == "list":
            items = store.expenses(month=args.month, category=args.category)
            for item in items:
                note = f"  # {item.note}" if item.note else ""
                print(
                    f"#{item.expense_id:<4} {item.day.isoformat()}"
                    f"  {item.category.value:<14}"
                    f" {format_money(item.amount):>12}{note}"
                )
            print(f"Total: {format_money(store.total(month=args.month))}")
        elif args.command == "report":
            print(format_report(store, month=args.month))
        return 0
    except (ExpenseError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
