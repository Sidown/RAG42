class test:
    def __init__(self):
        a = 'edisfjjwoi'
        b = 'iwoehjfiwhef'
        c = 'iejfpihepifjwe'
    
    def qqq(self):
        return self.a + self.b + self.c


def spell_reducer(spells: list[int], operation: str) -> int:
    ops = {"add": operator.add,
           "multiply": operator.mul,
           "max": max,
           "min": min}
    op = ops.get(operation)

    if not op:
        return "Use add, multiply, max or min for operation"
    return functools.reduce(op, spells)


def partial_enchanter(base_enchantment: callable) -> dict[str, callable]:
    fire = functools.partial(base_enchantment, 50, "fire")
    earth = functools.partial(base_enchantment, 50, "earth")

    return {"fire": fire,
            "earth": earth}


@functools.lru_cache
def memoized_fibonacci(n: int) -> int:
    if isinstance(n, int):
        if n < 2:
            return n
        return memoized_fibonacci(n - 1) + memoized_fibonacci(n - 2)
    else:
        return f"Error: {n} is not a number"


def spell_dispatcher() -> callable:

    @functools.singledispatch
    def dispatch(spell: any) -> str:
        return "Spell type not found"

    @dispatch.register(int)
    def _(spell: int) -> str:
        return f"casting spell with a power value of {spell}"

    @dispatch.register(str)
    def _(spell: str) -> str:
        return f"Casting {spell}"

    @dispatch.register(list)
    def _(spell: list) -> str:
        return f"casting {len(spell)} spells on target"

    return dispatch


def base_enchantment(power, element, target):
    return f"Enchanting {target} with {element} (power: {power})"